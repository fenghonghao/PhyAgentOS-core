# 琶洲 Track 2 复赛：机器人代码接口规格（原语 + skill）

日期：2026-11-03。状态：v1.1，用户已审定（15:27）：去掉 pick/place/stack_on skill，几何移进 locate，保留 handover。

> **11-03 21:00 赛制变更：单臂、12 题（目标物进容器 / 积木水果分类 / 数字等式）。** 现行设计见 §13；§3–§9、§11 里的双臂内容（`handover`、两臂并行、`arm_conflict`）是历史记录，代码仍兼容双臂但不再使用。

本文取代 `2026-11-03-paizhou-track2-skill-design.md`（设计稿 v2）§4–§8 的工具面与流程设计；v2 中的粗粒度 Action 工具、`plan.execute`、`deadline_s`、`plan.time_left`、按剩余时间 finalize 的恢复规则全部作废。

术语：**skill** 指一般意义上的技能（本文第 4 节的函数）；PhyAgentOS 的 Skill 包称 **paos skill**。

## 1. 总体结构

```text
LLM（DeepSeek，开思考）
  │  每次请求末尾追加 [time_left_s=…]
  ▼
robot.run(code)            ← paos skill 唯一暴露给 LLM 的 Action 工具
  │  在 skill_server 内执行 Python
  ▼
第 3 层  模型现写的任务逻辑（抓放顺序、选臂、并行编排、异常处理；抓放按 SKILL.md 示例用原语现写）
第 2 层  skill 函数：handover（预写、测过、可选）
第 1 层  原语 P1–P9（感知、运动、夹爪、状态；几何细节由 locate 算好）
  ▼
perception 节点 / motion_server ×2 / joint_trajectory_controller ×2 / piper 驱动 ×2
```

模型可以只用 skill，也可以绕过 skill 直接用原语。skill 不适用时返回失败原因，不悄悄做错。

## 2. 约定

- 坐标：世界系（桌面系），米；角度为度。两臂基座在世界系下的位姿来自标定。
- 臂：`"left"` / `"right"`。
- 失败：领域结果（没到位、没夹住、没找到）用返回值；驱动断连、急停、碰撞停机抛 `RobotFault`，中断整段代码交回模型。
- **禁止任何停止判断**：原语、skill、`robot.run`、SKILL.md、profile 配置中不得有超时停止或"时间不够就停"的逻辑。结束由 LLM 决定，超过 180 s 由裁判停机。skill 内部的"重试一次"是次数限制，不读时间。

## 3. 第 1 层：原语

### 3.1 工具 `robot.run(code)`

- 在 skill_server 独立线程执行；命名空间只含本文原语、skill、`numpy`、`math`。
- 命名空间在同一场景内持久，场景开始时清空。
- 返回：print 输出、traceback（如有）、本段原语与 skill 调用台账（参数、结果、起止时间）、用时。台账同时落盘，作为工具日志。

### 3.2 原语表

| # | 签名 | 返回 |
|---|---|---|
| P1 | `locate(text, camera=None, min_score=0.2, allow_move=True, frames=1)`；`text` 为字符串或同义说法列表 | `usable`、`found`、`center_xyz`、`top_z`、`size`(l,w,h)、`long_axis_deg`、`grasp_yaw_deg`、`grasp_xyz`、`open_width_m`、`views`、`score`、`alternatives`、`moved_arm`、`note` |
| P2 | `show(camera)` | 图片附进返回。**核实 PAOS 能把工具结果图片交给模型后开放** |
| P3 | `back_project(u, v, camera)` | 像素对应世界点。同 P2 |
| P4 | `move_to(arm, xyz, yaw_deg=None, tilt_deg=0, path="auto", wait=True)` | `reached`、`final_xyz`、`err_m`、`flipped`、`reason`∈{`ik_failed`,`joint_limit`,`arm_conflict`}；`wait=False` 返回句柄 |
| P5 | `home(arm, wait=True)` | 同 P4 |
| P6 | `wait(*handles)` | 各句柄结果；无超时 |
| P7 | `set_gripper(arm, "open"/"close", width_m=None, wait=True)` | `width_m`、`holding`、`effort` |
| P8 | `state()` | 每臂末端位姿、关节、夹爪开口、`holding`、`busy`、手中物体记录 |
| P9 | `time_left()` | 剩余秒数，只读；原语与 skill 内部不调用；SKILL.md 不写时间阈值 |

`usable` = 找到 且 算出 3D 且 `views ≥ 2`。

`locate` 负责抓取几何（只按物体尺寸算，与任务无关）：
- `grasp_xyz`：底面中心的 xy，z = `top_z` − d，夹持深度 d = min(物高/2, `grasp_depth_max`，默认 3 cm)。
- `open_width_m`：短边 + `open_margin`（默认 3 cm），不超过夹爪最大开口；短边 + 余量超过最大开口时 `note` 含 `too_wide`。
- `grasp_yaw_deg`：夹爪闭合方向垂直于长轴（夹短边）。

### 3.3 P1 `locate` 内部顺序（得到 `usable` 即停）

1. 锚点相机（默认头部）跑 SAM3；最高分实例为结果，其余入 `alternatives`。
2. 没找到：依次以两台腕部相机为锚点。
3. 依次尝试 `text` 中的其他说法；仍无则取新帧再试一次。
4. 算 3D：有深度用掩码点云（最小外接矩形、顶面取 95 百分位）；无深度用多视角视觉外壳；单视角只给底面，`note` 标高度未知。跨相机匹配：锚点估计投影到其他相机，选包含投影点的掩码。
5. 自动腕部复测：`views < 2` 或无 3D，且未指定 `camera`，且 `allow_move=True` 时，选**空闲、空手、观察位 IK 有解**的臂中离物体近的一只，把腕部相机移到物体上方 25 cm（扣除手眼标定的相机偏移）后融合重测。臂留在原地，`moved_arm` 写明。无可用臂则不移，`note` 说明。
6. 机械臂投影靠近掩码时 `note` 标"可能被遮挡"。

### 3.4 P4 `move_to` 内部

- `path`：`"auto"` 能走直线走直线、否则关节空间；`"line"` 只走直线，不行则不动报失败；`"joint"` 关节空间。
- IK 无解时，**仅在空夹爪时**自动试 yaw+180°（`flipped=True`）；手中有物体时不翻转，直接报 `ik_failed`。
- 双臂防碰撞：目标离另一臂末端或其在途目标过近时不动，报 `arm_conflict`。

## 4. 第 2 层：skill 函数

只保留 `handover`。抓、放、叠由模型按 SKILL.md 示例用原语现写（几何已由 `locate` 给出）；若仿真实测中模型经常写错，再补回 `pick`/`place`。

`handover(giver, receiver, at=None, mode="air")`，只用 P1–P8 实现，不读时间，返回 `ok`、`reason`、`holding`、实测位置。

| mode | 流程 | 默认值（均可覆盖） | 失败原因 |
|---|---|---|---|
| `air` | 递出方（须 `holding`）举到交接点 → 接收方从对侧水平靠近、张开、夹物体下半部 → **接收方 `holding=True` 后递出方才张开** → 递出方退出 | 交接点、两臂交接姿态、接收方倾角：YAML 默认值（两臂中间、桌面上方约 25 cm），明天真机标定 | `not_holding`、`receiver_empty`（递出方不松手，交回模型）、`unreachable` |
| `table` | 递出方放到交接点桌面 → 退出 → 接收方 `locate` 后抓起 | 交接点桌面坐标：YAML | 同上 + `not_found` |

为什么只保留它：难点在两臂相对姿态的标定与"确认接收方夹住才松手"的时序，模型现写最容易错，错了物体掉落；抓放的难点是几何，已移进 `locate`。

## 4.1 SKILL.md 抓放示例（在 sim-piper 测过后放入）

```python
def pick(arm, name):
    s = locate(name)
    if not s.usable:
        return None, s.note
    above = np.add(s.grasp_xyz, [0, 0, 0.08])
    move_to(arm, above, yaw_deg=s.grasp_yaw_deg)
    set_gripper(arm, "open", width_m=s.open_width_m)
    move_to(arm, s.grasp_xyz, path="line")
    g = set_gripper(arm, "close")
    move_to(arm, above, path="line")
    return (s if g.holding else None), g

def place(arm, xy, surface_z, s):
    d = s.top_z - s.grasp_xyz[2]                 # 夹持深度
    z = surface_z + 0.01 + (s.size[2] - d)       # 物体底面离支撑面 1 cm 时夹爪高度
    move_to(arm, [xy[0], xy[1], z + 0.08])
    move_to(arm, [xy[0], xy[1], z], path="line")
    set_gripper(arm, "open")
    move_to(arm, [xy[0], xy[1], z + 0.08], path="line")
```

叠放 = 重新 `locate` 底座取最新 `top_z` 作为 `surface_z` 再 `place`。

## 5. 第 3 层：留给模型

任务顺序、选哪只臂、两臂如何并行（`wait=False` + `wait`）、skill 不适用时用原语现写、失败后怎么办、何时结束。

## 6. PAOS 侧改动

- 每次模型请求在最新消息末尾追加 `[time_left_s=…]`：首次请求在 `agent/context.py` 的 `_build_message_context`，之后在 `add_tool_result`。追加在末尾以保住前缀缓存。截止时刻由 `run_scene.sh` 写入环境变量 `PAOS_SCENE_DEADLINE` 与文件，PAOS 读环境变量，skill_server 读文件；未设置时不注入。
- `forge_tool_start_action` / `forge_tool_action_result` 等到终态再返回，不设超时返回。
- `agents.defaults.maxToolIterations` 设为 512（schema 默认即 512；本机旧配置写的是 40，已于 11-03 改为 512）。
- paos skill 的 `gateway.yaml` 中 `tools.invoke_timeout_ms`：先确认是否会取消 Action；会则设到不可能触发。

## 7. 实现后端

| 部件 | 后端 | 待核实 |
|---|---|---|
| `robot.run`、skill | skill_server 节点（Python，Dora） | Gateway `invoke_timeout_ms` 语义 |
| P4 P5 P6 | 每臂一个 motion_server（registry）→ joint_trajectory_controller → piper 驱动；Action 客户端照 `motion_action_policy` | motion_server 是否支持笛卡尔直线；不支持则 1 cm 插值逐点 IK |
| P7 | 驱动 `action/gripper` 收稀疏 JointCommand；驱动 state 读开口 | piper_sdk 是否给夹持力；`holding` 阈值、夹爪最大开口需真机标定 |
| P1 P2 P3 | perception 节点：SAM3 + 深度或视觉外壳 | 相机有无深度、外参坐标系（评委问题 1）；SAM3 单帧耗时；腕部相机手眼偏移 |
| P8 P9 | 驱动状态；`run_scene.sh` 写的截止时刻 | — |

开发顺序（先在 MuJoCo `sim-piper` 上）：`robot.run` 骨架 + P8/P5/P4 → P7 → P6 双臂 → P1 → SKILL.md 抓放示例实测 → `handover` → 台账与 P9。

## 8. 待用户决定

1. `tilt_deg`：`handover(mode="air")` 需要接收方水平夹取，因此建议第一版保留非竖直姿态。
2. `handover` 默认 `air` 还是 `table`：取决于明天公布的成功条件是否要求空中传递。
3. P10 `objects()`（开计时前预扫描物体表）：取决于评委问题 8。
4. 是否加 `push(arm, obj, direction)` 处理夹不起来的物体：第一版不做。

## 9. 实现状态（11-03 晚）

代码：`example/piper-dual-tabletop/`（paos skill）；PAOS core 改动见 §6。

| 部分 | 状态 | 验证 |
|---|---|---|
| 原语 P1–P9、`handover`、执行器（含沙箱：内置函数白名单、import 白名单、拒绝双下划线） | 完成 | 46 个单测 |
| 假世界 + DeepSeek（开思考）离线测评 `dev/llm_trial.py` | 完成 | 加竖直限高后 14/15，唯一失败为交接任务 181 s 超时 |
| Dora 节点 `tabletop.dora_node`（Forge Action provider + 双臂 Dora Action 客户端 + 相机输入） | 完成 | MuJoCo 单臂全链路：Gateway → robot.run → motion_server/jtc/夹爪 |
| 自研 IK（解析雅可比 DLS，多种子，2 ms/次）+ MoveJoints；直线运动 2 cm 小步 MovePose | 完成 | 仿真 |
| `real-piper-dual` profile、`skill.yaml`（PAOS 严格校验通过）、`start.sh`、`run_scene.sh`、`scripts/make_bundle.sh` | 完成 | 打包 147 文件 |
| SAM3 分割（transformers 5.22.1，权重 ModelScope 镜像，按帧缓存视觉特征） | 完成 | 渲染场景 IoU 0.92–0.95；RTX 5060 每帧约 0.55 s，显存 2.1 GB |

实现中发现的约束：
- **竖直朝下限高**：PIPER joint5 ±70°，竖直朝下时 TCP 最高约在底座平面上方 0.14 m（URDF 采样）。高处位姿必须倾斜。`TOPDOWN_MAX_Z` 进配置与 SKILL.md，现场按底座安装高度复测。
- **motion_server 0.1.0** 只做 IK + 关节空间轨迹；其 IK 以当前关节为种子，从初始位转到竖直朝下会陷局部极小，因此远距离移动改用自研 IK + MoveJoints。
- **Dora 输入队列默认 10 条，溢出静默丢消息**：丢掉 invoke/result 请求会使 Gateway 1.0.2 的调用卡在 accepted、并发名额被占死（后续 robot.run 全部 503）。已给工具消息与运动/夹爪目标和结果设 `queue_size: 1000`；修复前压力测试第 3/6 次卡死，修复后 10/10 通过。
- 驱动启动即移到 `positions.initial`，关闭时移到 reset 位并闭合夹爪；joint3 初始须留 −0.02 rad 余量（编码器零漂越过上限 0 会使 IK 全部拒绝起点）。

## 10. 明天现场清单（按顺序，11-03 21:00 按单臂改写）

0. 防休眠：`systemd-inhibit --what=sleep:idle:handle-lid-switch --why="paizhou competition" sleep infinity &`（休眠会断开 CAN、相机与 Dora 数据流）。
1. **先问组委会**（§13.4 的 6 个问题），按答案改入口和配置。
2. 主机：`sudo apt install can-utils`；`sudo usermod -aG video $USER` 后重新登录；`./start.sh` 自检；确认 `~/.PhyAgentOS/config.json` 中 `agents.defaults.maxToolIterations` 为 512。
3. CAN：piper_sdk `can_activate.sh` 把机械臂接口命名为 `can0`（`profiles/real-piper/piper.yaml` 的 `port`）；`robots_agilex_piper test can0` 看夹爪开合。
4. 驱动初始位：确认 `profiles/real-piper/piper.yaml` 的 `positions.initial` 不碰立杆和挡板。
5. 标定由官方提供：头部相机外参、腕部手眼、内参 K 填入 `profiles/real-piper/skill_server.yaml` 的 `cameras`（取消注释；世界系 = 机械臂底座系；支持 4x4 矩阵、`T_inverse`、`wrist_frame: link6`）。量 `table_z`（桌面在底座系的高度）。两路相机的 `device_serial` 填进 `camera_head.yaml` / `camera_wrist.yaml`。用位置已知的物体核对 `locate` 误差。
6. 量工作区：挡板和立杆的位置填 `robot.workspace`（x/y 范围、z_max），`move_to` 越界返回 `outside_workspace`。
7. 复测 `TOPDOWN_MAX_Z`、夹爪最大开口 `max_opening_m`、`holding` 阈值。
8. `robot.run` 逐项：`state()`、`home()`、低处竖直 `move_to`、开合夹爪；`pick` / `place_at` / `place_in` 各试一次（`place_in` 看松手高度是否在容器沿上方 3 cm、容器有没有被碰）。
9. `look` / `inspect` / `orient`：在公开赛题的真实物品上各跑一次，看 `~/.PhyAgentOS/logs/piper-dual-tabletop/look/` 里存下的图和答案；数字太小就把 `camera_head.yaml` 彩色分辨率提到 1280×720/800（同时改 K）。
10. 确认读等式的方向，填 `robot.read_up_deg` / `read_right_deg`（默认裁判站在桌前、面对机械臂：字头朝 180°，从左到右是 +y）。
11. 调速：驱动 `motion_speed_percent`、`motion.velocity_scale`，在不丢精度前提下尽量快。
12. 用公开赛题按三类各跑全流程（PAOS agent），保存日志。
13. 改包后重装：同版本 `paos skill install --local` 不会覆盖（提示 already ready）。要先 `paos skill stop piper-dual-tabletop`、`paos skill remove piper-dual-tabletop`，再 `install --local dist/…tar.gz -y`；两个 venv 在包外，不受影响（11-03 已验证）。改了 `profiles/real-piper/` 后先跑 `python3 scripts/make_vla_profile.py real-piper` 同步 VLA profile。
14. B 方案已是独立的 paos skill `piper-vla`（`example/piper-vla/`），采集、训练、部署和受本机硬件限制的参数见其中的 `DESIGN.md`。

## 11. PAOS 全流程实测（11-03 18:07，sim-piper）与待定

`paos skill install --local` → `paos skill start --profile sim-piper` → `run_scene.sh`，任务：左臂竖直朝下到点、开合夹爪、回 home。结果 succeeded，TCP 误差 0.1 mm，总 143 s：

| 部分 | 耗时 |
|---|---|
| 4 段 robot.run 实际执行 | 29.5 s |
| 8 次模型请求（激活、查工具契约、建任务、4 次 start_action、最终回复），平均约 6 s/次 | 约 50 s |
| finalize 的验证模型调用（失败：返回空内容） | 约 60 s |

结论与已做处理：
- `deepseek-flash` 默认开思考（不带参数也返回 reasoning），PAOS 配置无需改动即满足"必须开思考"。
- PAOS 在 `skill start` 时自动运行包内 `start.sh`（传 `PAOS_SKILL_PROFILE`）：已改为 sim-* 跳过 CAN/相机检查、依赖按需求文件哈希按需安装。
- SKILL.md 写明 robot.run 契约，省掉 `forge_tool_context` 一轮。
- 验证模型失败原因：验证服务默认 max_tokens 2048，deepseek-flash 默认思考，推理耗尽额度后正文为空。

**待 11-04 实测后决定**：验证模式 `off`（推荐：finalize 立即按执行记录判定，省约 60 s，不受验证模型故障影响）还是 `audit` + 修验证模型（加大 max_tokens 或关思考）。定下后写进 SKILL.md 工作流程，不再由模型自选。

## 12. B 方案：学习策略（ACT / π0.5），11-03 晚实现

> **11-03 23:00 起作废**：B 方案已拆成独立的 paos skill `piper-vla`（`example/piper-vla/`，说明见其中的 `DESIGN.md`），不再嵌在 `robot.run` 里，也没有 `cmd_mux`、`real-piper-vla` profile 和 `make_vla_profile.py`。本节保留作记录。

### 12.1 结构

- 新 profile `real-piper-vla`（`scripts/make_vla_profile.py` 由 `real-piper` 生成，A 方案调参后重跑即可同步）：A 方案全部节点 + `cmd_mux` + `vla_policy`。模型仍只用 `robot.run`，多两个原语 `policies()`、`run_policy(name, seconds, task=None)`；B 方案可以整场顶替，也可以和 `locate` 等原语混用。
- 命令切换：轨迹/夹爪控制器在动作结束后仍持续重发保持目标，会和策略在驱动处打架。`cmd_mux` 在策略执行期间只放行策略命令；策略停止后，先把控制器重新锚定到当前位置（关节 MoveJoints 到当前值，夹爪目标 = 策略最后一次夹爪命令，保持夹紧力），再切回控制器，避免手臂弹回策略前的位置。
- 无停止判断：执行多久由模型给出 `seconds`；包内只有存活检查（策略节点停发状态时报 `RobotFault`）。
- 数据格式（采集与推理共用 `nodes/vla/src/piper_vla/spec.py`）：每臂 joint1..6 + gripper（总开口，米），单臂 7 维（双臂 14 维）；`action` 是下发的目标值；相机 head / wrist，RGB 320×240，30 Hz。
- 独立 venv `piper-dual-tabletop-vla`：lerobot 0.6.1、torch 2.11.0+cu128、transformers 5.5.4（与 skill_server 的 transformers 5.22.1 / numpy 2.5 冲突，故分开）。训练与推理用同一份依赖。
- π0.5 分词器：LeRobot 写死需授权的 `google/paligemma-3b-pt-224`。`setup_venv.sh` 用 ModelScope 的同名文件填 HF 缓存，运行时 `HF_HUB_OFFLINE=1`。

### 12.2 文件

| 路径 | 作用 |
|---|---|
| `nodes/vla/setup_venv.sh [--pi05]` | 建 VLA venv、分词器缓存、ACT 骨干权重；`--pi05` 再下 pi05_base（14.5 GB，ModelScope，校验 sha256） |
| `nodes/vla/collect.sh real\|sim --dataset N --task "..."` | 键盘/手柄遥控 + 录 LeRobot 数据集（操作界面带三路相机预览） |
| `nodes/vla/train.sh act\|pi05 N [lerobot-train 参数]` | 训练预设 |
| `nodes/vla/export_model.sh RUN NAME "描述" "task"` | 拷到 `~/.PhyAgentOS/models/piper-vla/NAME`，打印带 sha256 的配置条目 |
| `nodes/vla/collect/*.yaml` | 采集数据流模板 + 数据格式/遥控配置 |
| `nodes/vla/src/piper_vla/` | spec、teleop（IK）、ui（pygame）、recorder、collect_node、policy_node |
| `nodes/skill_server/src/tabletop/{policy_backend,mux_node}.py` | skill_server 侧客户端、命令切换节点 |
| `profiles/{real-piper,sim-piper}-vla/vla_policy.yaml` | 已训练策略清单（name / path / sha256 / description / task / n_action_steps） |
| `dev/gw_run.py PORT CODE` | 不经模型直接向 Gateway 发一段 robot.run |

### 12.3 明天流程

1. 工作站：`nodes/vla/setup_venv.sh --pi05`，然后 `nodes/vla/selftest.sh`（π0.5 的训练和推理只能在这里第一次验证；跑完删掉 `~/.PhyAgentOS/vla/selftest`）。可从笔记本拷 `~/.PhyAgentOS/models/pi05_base`、`~/.PhyAgentOS/models/paligemma-3b-pt-224-tokenizer`、`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` 省下载。
2. 采集（接机器人的那台电脑，先 `paos skill stop piper-dual-tabletop`，驱动独占 CAN）：`nodes/vla/collect.sh real --dataset <名> --task "<英文指令>"`。每条都从 H（回 home）+ 夹爪张开开始，回车开始/保存，退格丢弃，Esc 退出（数据集收尾），再 Ctrl-C。每个任务 30–50 条。
3. 传数据：`rsync -a ~/.PhyAgentOS/vla/datasets/<名> <工作站>:~/.PhyAgentOS/vla/datasets/`。
4. 训练：`nodes/vla/train.sh act <名>`（π0.5：`train.sh pi05 <名>`）。
5. 导出：`nodes/vla/export_model.sh <run_dir> <策略名> "<任务与场景，示教约 N 秒>" "<指令>"`，把输出贴进 `profiles/real-piper-vla/vla_policy.yaml`，模型目录放到比赛机同一路径。
6. 启动：`paos skill start piper-dual-tabletop --profile real-piper-vla`（`start.sh` 校验每个模型的 sha256），用 `dev/gw_run.py` 先手动试 `run_policy`。
7. 17:00 前封存代码、`vla_policy.yaml` 和模型文件。

### 12.4 本机实测（11-03 晚）

- 合成数据：录制、续录、分位数统计（π0.5 归一化要用）、ACT 训练 30 步、导出、加载推理全部通过。
- MuJoCo 全链路：自动操作录 3 条（各约 150 帧）→ ACT 训练 600 步 → `paos skill start --profile sim-piper-vla` → robot.run 里 `policies()` → `home` → `run_policy(4 s)` → `move_to`：策略 4 s 出 100 步（25 Hz），单步平均 5.9 ms；交还后 `move_to` 到位，误差 4 mm。
- 修过的性能问题：每步新建线程（CUDA 句柄重建）→ 常驻推理线程；CPU 上把图像转 float 要 44 ms → 先把 uint8 传上 GPU 再转，torch CPU 线程限 2 个。
- π0.5：分词器离线加载通过；本机无法跑 π0.5 训练和推理（建模型时先按 fp32 建，内存峰值超过 15 GB 被杀；8 GB 显存也装不下）。LeRobot 的 π0.5 在权重加载失败时只打印一句话、返回随机初始化模型，`load_policy` 已改为捕获这句输出并报错。
- 自检脚本 `nodes/vla/selftest.sh`：合成 14 维数据集 → ACT 200 步 → π0.5（从 pi05_base）20 步 → 用策略节点自己的加载代码各推理一次，打印加载时间、显存、单步耗时。本机 `selftest.sh act` 通过（ACT batch 16 约 4 步/秒，推理单步 0.5–10 ms）。

### 12.5 受本机硬件限制的参数（明天在 4090 工作站上放开）

| 参数 | 位置 | 本机取值 | 为什么 | 工作站上 |
|---|---|---|---|---|
| π0.5 推理 | `profiles/real-piper-vla/vla_policy.yaml` 的策略条目 | 不启用（8 GB 显存装不下 3B 模型 + SAM3） | 显存 | 启用；比赛机须是工作站（约 8 GB 显存给 π0.5，2 GB 给 SAM3） |
| 策略节点 CPU 线程 | `profiles/real-piper-vla/vla_policy.yaml` 的 `torch_threads` | 2 | 本机负载高时 CPU 线程互抢（不限时单步 44 ms） | 4–8 |
| 录制图像写盘线程 / 视频编码线程 | `nodes/vla/collect/real-piper.vla.yaml` 的 `dataset.image_writer_threads` / `encoder_threads` | 4（每相机）/ 2 | 笔记本 CPU | 8 / 4（若采集机是工作站） |
| ACT 训练 batch / 步数 | `nodes/vla/train.sh` act 预设：16 / 30000 | 本机只做过 4–8 / 30–600 步的冒烟 | 冒烟测试 | 预设已按 4090 设；显存够可 `--batch_size=32`（步数可相应减） |
| π0.5 训练 | `train.sh` pi05 预设：bf16、梯度检查点、只训动作专家、batch 8、3000 步 | 本机无法训练 | 24 GB 显存是 4090 的上限，不是笔记本 | 4090 上保持；只有 ≥48 GB 卡才可去掉 `train_expert_only` |
| SAM3 精度 | `profiles/real-piper/skill_server.yaml` segmenter `dtype: float16` | float16 | 本机速度（约 0.55 s/帧） | 保持 float16；实测单帧耗时后再定 |
| `locate` 默认帧数 | `primitives.py` `frames=1`（SKILL.md 同） | 1 | SAM3 在本机慢 | 若工作站单帧 <0.2 s，可把默认改 2（多帧融合更稳），同步改 SKILL.md |
| 图像分辨率 | `nodes/vla/collect/real-piper.vla.yaml` 与 `vla_policy.yaml` 的 cameras.size | 320×240 | 训练速度（按 7 小时窗口选） | 不必放开：π0.5 内部缩到 224×224；ACT 用 640×480 训练约慢 3 倍。若要改，必须在采集前改且两处一致 |

配置文件里这些项都标了 `HARDWARE-LIMITED` 注释，可 `grep -rn HARDWARE-LIMITED example/piper-dual-tabletop` 一次找全。

### 12.6 风险

- 没有主臂：示教用键盘/手柄遥控末端，动作偏慢偏僵；遥控方向、30 Hz 流式命令下驱动的跟随、夹爪行程都还没在真机验证。
- 采集、训练、测试都挤在 7 小时内，还要和 A 方案抢机器人。
- 示教从 home + 夹爪张开开始，比赛时 `run_policy` 前必须同样起步（SKILL.md 已写）。

## 13. 单臂赛制（11-03 21:00 起）

### 13.1 新规则要点（新版手册 HTML，落款 11-03）

- 12 题：6 道公开（赛前训练调试）+ 6 道保密（正式计分，每题 100 分）；每组 3 简单 + 2 中等 + 1 困难。
- 简单"目标物品搬运"：每题给目标图像，把 1–2 件物品放进对应容器。完整进容器且稳定 100，容器外离边缘 ≤10 cm 得 50；每碰动一次干扰物 −10，每碰动一次容器 −50。
- 中等"物品分类整理"：积木和水果，1–2 件散置；指令"将桌上积木和水果分类摆放好。"；得分 = 待归类件数减少的比例。
- 困难"数字逻辑摆放"：3 个数字块 + 加减乘除等号各 1，用全部数字、1 个运算符和等号摆成"数 符 数 = 数"或"数 = 数 符 数"；结构 50 + 成立 50。
- 手册没写：PAOS / Skill 包、臂数、相机、任务如何下发、完成信号形式、执行时限。现场照片：一条 PIPER 装在桌后沿底板左端，底板中间立杆顶有一台深度相机，腕部有相机，工作区三面白色挡板。

### 13.2 代码改动

| 改动 | 位置 | 说明 |
|---|---|---|
| 单臂接口 | `primitives.py` `namespace` / `_bound` | 只配一条臂时 `move_to(xyz)`、`set_gripper("open")`、`home()`、`state()` 都不带臂名（习惯性写的臂名被忽略）；没有 `handover` |
| `look()` | `vision.py`、`primitives.py` | 当前相机画面（可画编号圈、可附参考图、`zoom=True` 附放大格子）→ deepseek-flash（OpenAI 兼容接口）。key 取环境变量或 PAOS 配置，不打印。**关思考**：开思考时一张图几分钟不返回，关掉约 1.3 s。无超时，传输错误按次数重试 |
| `inspect()` / `orient()` | 同上 | 空夹爪的腕部相机逐个移到物体上方近拍，按块边转正拼成一张图问一次；`orient` 给每块出 4 个转向副本，问哪张是正的，换顺序问 3 次投票，并带上已识别的字符（防 6/9 混淆） |
| `heading_of()` | `primitives.py` | 图像方向 → 世界朝向；`camera="tile"` 用于转正后的格子 |
| `locate_all()`、`locate(near=)` | `primitives.py` | 头部相机一次列出全部实例；`near` 锁定某个实例再精测（腕部复测也锁同一实例） |
| `pick` / `place_at` / `place_in` | `skills.py` | 竖直下降、竖直抬起；松手只再张开 1.5 cm（不推邻物）；`place_in` 在容器中心、物体底部高出容器沿 3 cm 处松手，夹爪不进容器，够不到时依次试 0/−30/−45° 倾角 |
| 工作区边界 | `robot.workspace` | 越界返回 `outside_workspace`（挡板、立杆） |
| 阅读方向 | `robot.read_up_deg` / `read_right_deg` | 等式题用；默认裁判站在桌前、面对机械臂 |
| profile | `profiles/real-piper`（新） | 双臂 profile 移到 `dev/legacy/`（不进包） |
| B 方案 | `example/piper-vla/`（11-03 23:00 拆出） | 独立 paos skill，A 方案里不再有任何 VLA 代码、配置和扩展口；见该目录 `DESIGN.md` |
| SKILL.md | — | 单臂 API、影响写法的计分规则、三类题示例代码 |

### 13.3 验证（11-03 晚）

- 单测：A 方案 77 个通过（11-04 00:10）。覆盖单臂接口、`locate_all`/`near`、`pick`→`place_in`、工作区、`look` 请求内容、`inspect`/`orient`、沙箱逃逸用例、启动死锁复现、取消与硬件故障的区分，以及 SKILL.md 三份手册的参考代码原样在假世界里跑通（`look`/`inspect`/`orient` 用真值替身）。B 方案的测试见 `example/piper-vla/DESIGN.md`。
- PAOS 链路：重装后 `sim-piper` 经 Gateway 跑单臂 `robot.run`：`move_to(xyz)` 误差 5.4 mm，带臂名的旧写法 1.8 mm，夹爪正常。
- deepseek-flash 真调用（假世界渲染的 8 个数字/符号块，头部 640×480）：整图画圈读字符 5/8；`inspect` 腕部近拍转正后 8/8；`orient` 两次运行都是 8/8，约 2 s。
- 模型照 SKILL.md 写代码（`dev/llm_trial_booth.py`，deepseek-flash 开思考，假世界，看图用真值替身；时间 = 模型耗时 + 按每次移动 2.5 s 估的执行时间）：两轮 13 次成功 12 次。目标搬运 1 件 2/2（约 35–40 s），2 件 3/3（55–67 s），分类 4/5（65–75 s），等式 3/3（155–172 s）。分类失败那次是把 8 件全部重排（估 194 s，超时），SKILL.md 加了"只动散置的 1–2 件"后 3/3。
- 等式题最慢：5 块 × 每块抓放约 20 s。`inspect` 顺带精测后 `pick` 不再二次飞过去（每块省约 4 s）；真机还要靠提速（`motion_speed_percent`、`velocity_scale`、`approach_m`）。时限确定后再看要不要让一块留在原地当锚点。

### 13.4 明早先问组委会

1. 任务怎么下发：自然语言是不是发给 PAOS（`paos agent`）？目标图像以什么形式给（文件路径 / 附在消息里 / 屏幕 / 桌上实物）？
2. "完成信号"具体是什么，`forge_task_finalize` 算不算？
3. 设备接口：能否直接接 PIPER 的 CAN 和相机 USB，还是必须用组委会的接口？
4. 相机型号与分辨率、内外参（相对哪个坐标系）何时给；执行时限多少；公开赛题的实物什么时候能拿到。
5. 等式从哪一侧读、字的朝向要求、"清晰可读"的容差。
6. 分类题的"聚集距离"、干扰物"移动"的判定阈值。

### 13.5 风险

- 头部相机 640×480 下数字只有十来个像素，所以读字用 `inspect` 腕部近拍（每块约 2 s 移动）；`orient` 只在渲染字体上测过。
- 目标图像对应、积木/水果判断全靠 deepseek-flash 看图，真实物品未测。
- 碰容器扣 50：`place_in` 的松手高度依赖头部深度相机测的容器沿高度，真机先试。
- `look` 不设超时（按"skill 包里禁止超时停止"）；网络卡住时这一段代码会一直等，直到裁判停机。429/5xx/空回答按次数重试 5 次（间隔 1、2、4、8、8 s）。
- RoboDojo 复测报告：deepseek-flash 关掉思考后成功率从 64% 掉到 20%（多数用完调用预算），所以主模型保持思考开启（PAOS 不设 `reasoningEffort` 时 DeepSeek 默认开）；`look` 只做看图识别、关思考，是另一回事。

### 13.6 经验积累：任务手册（PyRUA-RSI 思路，写成文字）

- 题型固定（三类），所以按 PyRUA-RSI 的思路积累"每类题怎么做"，但**写成 SKILL.md 里的文字手册**（Playbook A/B/C：计分要点、步骤、判断点和坑、参考代码），不做成包内函数：保密题和公开题同类不同细节，函数等于对公开题过拟合；文字 + 参考代码让模型照着改，保留 PyRUA-Lean"模型现场写代码"的机制。包里的代码只放任务无关的原语和 skill。
- 验证（11-03 晚，假世界 + 真值看图）：手册格式下 12 次成功 11 次，含三道变体题（换物体和容器、前后分组、换数字）5/6；失败那次是沙箱把 `xy.mean(axis=0)` 内部的 numpy 延迟导入拦了，已修（AST 拒绝带 `_` 段的 import；运行时对已加载的 numpy 子模块只返回安全视图）。
- 11-04 用公开题积累，经验总结由我（Claude）做：
  1. 关掉 PAOS evolution（`~/.PhyAgentOS/config.json` 的 `agents.evolution.enabled: false`）。它会在验证成功后自动写经验条目，满 3 次还会写进 SKILL.md 的托管区块，未经审阅，也会让封存版本在比赛中变化。
  2. 每跑一道公开题：`./run_scene.sh pub-<n> "<题目>" [目标图像]`，然后 `python3 dev/scene_digest.py` 出一页摘要（题目、每段代码和输出、`look`/`inspect`/`orient` 的问答、失败的原语调用、各原语耗时、发给看图模型的图）。
  3. 按摘要改对应手册：步骤、坑、参考代码（例如某物品 SAM3 要用的英文词、容器沿高度、夹取深度、摆放间距）；改完跑 `tests/test_skill_examples.py`（手册参考代码原样在假世界里跑）和全部单测，`make_bundle.sh`，`paos skill remove` + `install`。
  4. 17:00 封存前冻结 SKILL.md。

## 14. 代码审查（11-04 00:00，独立 agent，只看代码不评可行性）

已修（A 方案）：

- 启动死锁：腕部相机图像在机械臂第一帧关节状态之前到达时，`on_color` 在 Dora 主线程上等关节状态，而关节状态也由这个线程接收，节点卡死。现在没有关节状态时丢弃腕部图像（`perception.py`、`dora_backend.has_state`），有复现测试。
- 取消被当成硬件故障：夹爪控制器对取消回 `CANCELED`，原来被判为故障。现在 `gripper_fault()` 不把取消算故障；执行器在取消生效期间收到的 `RobotFault` 一律报"已取消"，取消后正常结束的代码块也标记为已取消。

按用户决定不修（记录在此）：

| # | 发现 | 状态 |
|---|---|---|
| 2 | 沙箱可借 `functools.update_wrapper` 把内部模块的全局变量合并进命名空间，执行系统命令 | 不修 |
| 3 | 三处等待不检查取消：看图 HTTP 请求、相机取帧、等控制器回结果；对方不回时代码块永不结束，之后的 `robot.run` 一直返回 BUSY | 未定 |
| 5 | `RobotFault` / `Cancelled` 是普通异常子类，模型代码 `except Exception` 能吞掉；硬件故障目前只在运动 / 夹爪指令的结果码里发现（`NO_FRESH_ROBOT_STATE`、`TRAJECTORY_EXECUTION_FAILED`、`CONTROLLER_UNAVAILABLE`，夹爪非成功码） | 未定 |
| 6–8 | 三份手册参考代码在找不到空位 / 空行 / 容器时会抓着物体出错 | 不修：流程待重新设计，现有手册只作参考 |
| 9–10 | B 方案：home 与 run 互相打断时不报错；`collect.sh` 多个 `--task` 被合并 | 不修（B 方案暂不处理） |

审查里的小问题和低置信度条目（未修）：`dora_backend` 未用的分段代码与过时注释；`inspect` 的 `notes` 没交给模型；执行器不限打印长度；A 方案 `gateway.yaml` 里 `robot.run` 的描述仍写双臂和 handover；SKILL.md 写 `usable` 需两视角而代码允许单个腕部俯视视角；手册 A、B 先看图后 `home()`；`download_sam3.sh` 没加 `curl -f`；`calib_compose.py` 是双臂遗留；上一场夹着东西时下一场 `pick` 会拒绝；图像编码（bgr8、16UC1 深度）被忽略；剩余秒数的环境变量优先于每场文件；`run_scene.sh` 不取消上一段仍在运行的代码；`orient` 在看不到物体时会抛错而不是返回 None。

用户对原语设计的判断（11-04 00:07）：整体太冗长，过度设计、防御性设计，待重构。

