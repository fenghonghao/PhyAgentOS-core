# B 方案说明：piper-vla（学习策略控制单臂 PIPER）

更新：2026-11-03 晚。独立的 paos skill 包，不依赖 A 方案（`piper-dual-tabletop`）的任何代码或配置。两个包用同一条机械臂，同一时间只能启动其中一个（A 的 Gateway 端口 19002，B 的 19003）。

## 1. 定位

用遥操作录示教数据，训练端到端视觉运动策略（ACT 或 π0.5，基于 LeRobot 0.6.1），比赛时由模型选择策略和运行时长。策略只会做训练过的那类任务，换了物体或布局往往不灵；所以 B 方案适合"公开题与保密题物品、布局接近"的情况，优先级低于 A 方案。

## 2. 架构与数据流

```
PAOS agent ──forge_tool_query / forge_tool_start_action──▶ Gateway :19003
Gateway ◀──▶ vla_server（Forge 工具端点 vla.server：policies / state / home / run）
vla_server ──policy_cmd（JSON：start / stop / home / status）──▶ vla_policy
vla_policy ──policy_status（JSON，约每 0.5 s，状态变化时立即）──▶ vla_server
vla_policy ──关节 + 夹爪目标（JointCommand，每个控制周期）──▶ PIPER 驱动（仿真时为 mujoco_sim）
头部相机、腕部相机（仿真：top / angle / hand）──图像──▶ vla_policy；驱动 ──关节状态──▶ vla_policy
```

- 没有 motion_server、轨迹控制器和指令切换：策略节点是驱动唯一的指令来源；不发指令时驱动保持最后一个目标。
- `vla_policy`：加载 `vla_policy.yaml` 里的策略（后台加载、校验 sha256）；推理在一个常驻线程里做（CUDA 句柄按线程建立，每步新建线程会慢几十毫秒），图像先以 uint8 送上 GPU 再转浮点；每步关节变化限幅（`max_joint_step_rad`，默认 0.12 rad），夹爪限在 0 到最大开口之间。
- 回初始位（`home`）：从当前关节沿 smoothstep 插值到示教起始姿态并张开夹爪，时长 = 最大关节角差 / `home_joint_speed`（默认 0.5 rad/s），至少 1 s。
- 节点从不自行结束一次运行：运行时长由模型给的 `seconds` 决定，取消立即停。

## 3. 给模型的工具（完整说明在 SKILL.md）

| 工具 | 类型 | 参数 | 结果 |
|---|---|---|---|
| `vla.policies` | 查询 | 无 | 每个策略的 `name`、`state`（ready / loading / error）、`description`、`task` |
| `vla.state` | 查询 | 无 | 每条臂的 `joints`（rad）、`gripper_width_m`，正在进行的 `run` / `home` |
| `vla.home` | 动作 | 无 | `reached`、`error`、`elapsed_s` |
| `vla.run` | 动作 | `policy`、`seconds`、可选 `task` | `steps`、`rate_hz`、`infer_ms`、`error`、`cancelled`、`gripper_width_m` |

典型流程：`vla.policies` 选描述匹配的策略 → `vla.home` → `vla.run`（`seconds` 约等于一条示教的时长）→ `vla.state` 看结果 → 再跑一次或换下一个策略 → `forge_task_finalize`。策略还在加载时 `vla.run` 会等它加载完；策略节点停止发布状态时工具返回 `POLICY_NODE_DOWN`。

## 4. 数据格式（采集、训练、推理共用 `nodes/vla/src/piper_vla/spec.py`）

- 状态与动作：每条臂 `joint1..joint6`（rad）+ `gripper`（两指总开口，m），单臂 7 维。`action` 是下发的目标值，不是测量值。
- 图像：真机头部 `head`、腕部 `wrist` 两路彩色，存成 320×240；仿真 `top`、`angle`、`hand` 三路。
- 频率 30 Hz。`nodes/vla/collect/<profile>.vla.yaml`（采集）和 `profiles/<profile>/vla_policy.yaml`（推理）的 `fps`、`arms`、`cameras` 必须一致（有测试检查）。
- 数据集位置：`~/.PhyAgentOS/vla/datasets/<名>`（LeRobotDataset，可续录；统计量含 q01/q99 分位数，π0.5 需要）。

## 5. 采集

```bash
paos skill stop piper-vla                      # 驱动独占 CAN，采集前停掉 skill
nodes/vla/collect.sh real --dataset cup --task "put the cup in the bowl"   # 仿真：collect.sh sim
```

- 操作台显示相机画面和状态。键盘（按住；Shift 慢速）：W/S、A/D、R/F 沿 x、y、z 移动，Q/E 偏航，T/G 俯仰，Z/X 夹爪开合，Space 夹爪全开/全合，H 回初始位，P 暂停，Enter 开始录 / 保存，Backspace 丢弃本条，F1–F9 选任务文本，Esc 退出（数据集收尾）。也支持 Xbox 手柄。
- 没有主臂，只能键盘或手柄遥操作（末端速度控制 + IK）。每条示教都从 H 回初始位、夹爪张开开始；不在初始位开始会提示。
- 只有关节状态和所有相机都新鲜（状态 ≤0.3 s、图像 ≤0.5 s）的帧才会记录。
- 每个任务录 30–50 条。

## 6. 训练、导出、部署

```bash
nodes/vla/setup_venv.sh [--pi05]               # VLA venv（带 hash 的依赖）、PaliGemma 分词器缓存、ACT 骨干；--pi05 下载 pi05_base（ModelScope，校验 sha256）
nodes/vla/train.sh act cup                     # ACT：chunk 50，batch 16，30k 步（4090 上约 1 小时）
nodes/vla/train.sh pi05 cup                    # π0.5：从 pi05_base 起，bfloat16，梯度检查点，只训动作专家，batch 8，3k 步（24 GB 显存）
nodes/vla/export_model.sh <run_dir> cup "<任务、场景、示教约 N 秒>" "<指令>"
# 把打印出的条目贴进 profiles/real-piper/vla_policy.yaml，然后打包、remove + install、start
```

- 训练输出：`~/.PhyAgentOS/vla/runs/<数据集>-<策略>-<时间>/checkpoints/last/pretrained_model`；导出到 `~/.PhyAgentOS/models/piper-vla/<名>`，条目带 sha256，`start.sh` 启动时逐个校验。
- π0.5 的分词器写死为需授权的 `google/paligemma-3b-pt-224`；`setup_venv.sh` 用 ModelScope 的文件填好 HF 缓存，运行时设 `HF_HUB_OFFLINE=1`。LeRobot 加载 π0.5 权重失败时不会报错、会返回随机初始化的模型；推理节点检查了这种情况，遇到就报错。
- 工作站第一件事：`nodes/vla/selftest.sh`，用合成数据把 ACT 和 π0.5 的训练、加载、推理各走一遍，打印加载时间、显存和单步延迟。

## 7. 目录

| 路径 | 内容 |
|---|---|
| `SKILL.md` | 模型读的文档：工具与工作流 |
| `skill.yaml` | manifest：`real-piper`、`sim-piper`；锁定的注册表节点（gateway、PIPER 驱动、mujoco_sim） |
| `profiles/real-piper/` | 数据流、Gateway 工具定义、驱动、两台相机、`vla_policy.yaml` |
| `profiles/sim-piper/` | 数据流、Gateway 工具定义、MuJoCo 配置、`vla_policy.yaml`（含冒烟测试策略 `sim_move`） |
| `nodes/vla/src/piper_vla/` | `tool_server.py`（Forge 工具端点）、`policy_node.py`（推理与回位）、`collect_node.py` + `recorder.py` + `teleop.py` + `ui.py`（采集）、`spec.py`（数据格式）、`kinematics.py`（遥操作 IK）、`endpoint_lease.py` |
| `nodes/vla/src/forge_tool/` | Forge 工具端点库的内置副本（来源见其中 VENDORED.md） |
| `nodes/vla/*.sh` | `setup_venv.sh`、`collect.sh`、`train.sh`、`export_model.sh`、`selftest.sh` |
| `nodes/vla/tests/` | 14 个测试：数据格式、遥操作、工具（运行、取消、等加载、节点掉线、回位）、策略节点回位 |
| `start.sh` | 启动钩子：VLA venv、节点 import、策略权重与 sha256、真机的 CAN 口（读 `piper.yaml`）、相机权限和本次运行的固定曝光（由头部相机自动曝光测得，倍率见环境变量 `EXPOSURE_SCALE`，默认 1.47） |
| `dev/` | `gw_call.py`（直连 Gateway 调一个工具）、`legacy/`（旧双臂 VLA 配置）；不进包 |

## 8. 验证（11-03 晚）

- 测试 14 个通过。
- PAOS 全链路（MuJoCo，策略 `sim_move`）：`vla.policies` 返回 ready；`vla.home` 1.0 s 到位；`vla.run` 4 s 出 101 步，25.2 Hz，单步推理 4.3 ms；`vla.state` 返回初始位关节和张开的夹爪。
- 本机此前还验证过：仿真自动遥控录 3 条示教 → ACT 训练 600 步 → 推理。
- 未验证：π0.5 的训练和推理（本机 8 GB 显存、15 GB 内存都不够，须在 4090 工作站上用 `selftest.sh` 验）；真机遥操作手感；驱动对 30 Hz 连续指令的跟随；示教数据量够不够泛化到保密题。

## 9. 受本机硬件限制的参数（上 4090 工作站再放开）

| 参数 | 位置 | 本机值 | 工作站 |
|---|---|---|---|
| π0.5 推理 | `vla_policy.yaml` 的策略条目 | 不启用 | 启用；比赛机须是工作站 |
| 推理节点 CPU 线程 | `vla_policy.yaml` 的 `torch_threads` | 2 | 4–8 |
| 录制写图线程 / 视频编码线程 | `nodes/vla/collect/*.vla.yaml` 的 `dataset.image_writer_threads` / `encoder_threads` | 4（每相机）/ 2 | 8 / 4 |
| 图像分辨率 | `cameras.size` | 320×240 | 不建议放开：π0.5 内部缩到 224×224，ACT 用 640×480 训练约慢 3 倍 |

`grep -rn HARDWARE-LIMITED .` 可找全。

## 10. 风险

- 没有主臂，示教只能键盘或手柄遥操作，动作偏慢偏僵，数据质量受影响。
- 策略对新物体、新布局泛化差；保密题与公开题细节不同时可能失效。
- 采集、训练、测试都要挤进调试日，还要和 A 方案抢机器人。
- 用 π0.5 时比赛必须在 4090 工作站上跑。
