# 琶洲算法大赛 Track 2 复赛 Skill 包设计

日期：2026-11-03。状态：v2，已吸收 builder check 意见，待用户确认第 11 节问题。

## 1. 意图与背景

11 月 5 日上午真机比赛：组委会提供统一的 PIPER 双臂构型设备、相机、桌面与物体。每队跑 3 个等难度场景，每场景启动一次、执行上限 180 秒，场景类型取自抓取放置、物品交换、堆叠整理，正式任务清单与成功条件赛前公布。交付物是整套 PhyAgentOS 加一个可安装运行的 Skill 包，组委会检查 Skill 包。

11 月 4 日 09:30–16:30 是唯一的真机调试窗口，17:00 封存。今天（11 月 3 日）的目标：把 Skill 包写完、在本机 MuJoCo 上跑通工作流、把真机驱动和相机依赖装好，明天到场只做接线、标定核对和参数调整。

方案思路来自 PyRUA-Lean：LLM 不逐步调工具，而是一次决策驱动一段带检查和重试的程序。PAOS 的 AgentLoop 是逐工具调用的，所以我们把"程序段"下沉为粗粒度 Action 工具，LLM 只在段与段之间做决策。

## 2. 目标、非目标、反目标

**目标**
- 三类场景都能在 180 秒内完成，名义路径 LLM 轮次不超过 6 轮（见 §4 的轮次账）。
- 每个 Action 工具自带结果确认与一次内部重试，返回结构化失败原因；`SKILL.md` 规定每种失败原因的下一步。对应评分"闭环执行 20 分"。
- 一条命令安装、一条命令启动、失败有明确退出码与日志；急停或驱动断连时节点自己停下并报告。对应"工程与稳定性 20 分"。
- 每场自动落盘：场景编号、版本摘要、起止时间、工具日志、连续执行录像、AgentTask 证据。对应手册"结果须保存"。
- 同一 Bundle 提供 `sim-piper` profile，今天在 MuJoCo 上验证整条 LLM → 工具 → 执行链。
- 第二只臂接不通时可降级：`real-piper-single` profile 只起左臂，抓放与堆叠照常，交换任务退化为单臂三段。

**非目标（今天不做）**
- 双臂同时运动。v1 两臂串行。
- 6D 位姿估计。只做 top-down 抓取，用足迹矩形长轴推 yaw。
- VLA 策略、离线兜底模式（比赛允许联网，用户已决定不做兜底）。
- 修改任何已发布 Node 二进制。IK、轨迹控制器、Gateway 复用 registry 现有节点。

**反目标（做了也算失败）**
- LLM 直接看到或发送关节轨迹。
- 工具结果不带失败原因、把失败包装成成功。
- 为了省时间绕过 PAOS 的 AgentTask 绑定或 Action 语义（例如把运动做成 Query）。
- 硬编码任何场景坐标。

## 3. 硬约束（来自手册与 PAOS）

| 约束 | 来源 | 影响 |
|---|---|---|
| 每场景启动 1 次、180 秒上限、计时从 Skill 收到任务开始 | 手册六(四) | 启动自检必须在收到任务前完成；工具带 `deadline_s` |
| 自主重试允许，人工重启不算 | 手册六(四) | 重试逻辑放在工具和 `SKILL.md` 里，进程不得崩 |
| 联网服务须申报并验证可用性、日志、替代方式 | 手册七(一) | LLM API 调用日志随包提交；"替代方式"写第二家 API，不做离线 |
| 保存场景编号、版本摘要、起止时间、工具日志、连续录像 | 手册六(六) | recorder 节点 + AgentTask evidence |
| 提交文件按"赛道_队伍编号_队名_版本号"命名，附文件列表与 SHA-256 | 手册二(四) | 打包脚本输出清单 |
| 封存后只允许裁判批准的兼容性修复 | 手册二(四) | 参数全部放 profile YAML，现场调参不改代码 |
| `required_tools` 既是健康检查集合也是 AgentTask 工具白名单 | PAOS Bundle 规范 §4 | 只有 LLM 可见工具进白名单 |
| 证据只在首个 Action 前和 finalize 时采集 | PAOS 开发者手册 §8 | 中间观测靠工具返回体 |
| Dora CLI 0.4.1、Gateway 1.0.2、`dora-message` 0.7.0 | PAOS 开发者手册 §9 | 新节点用同版本 `dora` Python 包 |
| Registry 不开放外部提交；Bundle 归档拒绝符号链接 | 开发手册 §4、SKILL_DEVELOPMENT | 自研节点以包内 Python 源码形式存在，venv 由 `start.sh` 在包外建 |
| `ForgeToolClient` 单次 HTTP 超时 `forge.requestTimeoutS`=10 s | PAOS 配置参考 | 阻塞等待只能在 PAOS 工具层循环轮询实现，不能让 Gateway 阻塞 |
| `audit` 是 `forge_task_create` 的 `verification.mode` 参数；audit 下任一执行记录非 `succeeded` 则任务 FAILED | `forge/task.py:1499` | 领域失败（空抓等）以 `succeeded` 加 `ok:false` 返回，只有驱动错误和碰撞停机才返回 Gateway `failed` |

## 4. 时间与轮次预算

单臂一次抓放周期（接近、下降、闭合、抬升、搬运、下降、松开、退出）按 motion_server 默认限速估 12–15 秒，真机首日保守取 20 秒。

| 场景 | 周期数 | 执行 | 感知 | LLM | 合计 | 余量 |
|---|---|---|---|---|---|---|
| 抓取放置 1 件 | 1 | 20 s | 3 s | 10 s | ~33 s | 147 s |
| 物品交换单臂 | 3 | 60 s | 5 s | 10 s | ~75 s | 105 s |
| 堆叠整理 4 件 | 4 | 80 s | 8 s | 10 s | ~100 s | 80 s |
| 堆叠整理 6 件 | 6 | 120 s | 12 s | 12 s | ~145 s | 35 s |

LLM 轮次账（每轮一次模型调用，DeepSeek-flash 关思考约 1 秒，开思考约 1.7 秒）：

| 轮 | 调用 |
|---|---|
| 1 | `activate_skill` |
| 2 | `forge_task_create`（`verification.mode=audit`） |
| 3 | `scene.describe` |
| 4 | `plan.execute(steps=[…])`，`wait_ms` 到位，一次返回整段结果 |
| 5 | `scene.locate` 确认终态 |
| 6 | `forge_task_finalize` |

名义 6 轮。每次失败恢复加 1–2 轮（`arm.*` 单步重试）。没有 `wait_ms` 时每个 Action 要 2 轮，6 件堆叠会涨到 18 轮，所以 §9 的 PAOS 改动是必做项。

六件以上的整理任务只剩一次重试余量，对策：
1. 规划一次，名义路径不回 LLM；只有工具返回失败才回 LLM 重选。
2. 每个 Action 接受 `deadline_s`，内部重试受剩余时间约束；`plan.time_left` 让 LLM 知道还剩多少。
3. 现场逐步上调驱动 `motion_speed_percent` 与 motion_server `velocity_scale`，只是 YAML 参数。
4. 堆叠时只对"栈顶"重新 locate，不对全场景重扫。

## 5. 闭环流程

```text
裁判下令
  └─ 队员执行 paos agent -m "<任务文本>"     # 入口脚本 run_scene.sh 封装：记场景编号、开始时间
       ├─ activate_skill(piper-dual-tabletop)
       ├─ forge_task_create(goal, criteria=子目标列表, verification.mode=audit)
       ├─ scene.describe(names=[任务文本中的物体])
       ├─ LLM 规划子目标序列，转成 plan.execute 的 steps
       ├─ forge_tool_start_action(plan.execute, wait_ms=170000)
       │     ├─ 全部成功 → 下一步
       │     └─ 某步失败 → 返回已完成步数 + reason + observed，LLM 按恢复表用 arm.* 单步补救
       ├─ scene.locate(最后目标) 确认终态
       └─ forge_task_finalize                   # 后置证据 + audit verdict
```

工具返回体统一带：`ok`、`reason`（枚举）、`elapsed_s`、`observed`（动作后实测：夹爪开口、末端位置、目标物体是否移动）。`ok:false` 仍以 Gateway `succeeded` 终态返回，见 §3。

## 6. 工具面（Gateway ToolSpec）

Gateway 只挂一个 provider：`skill_server`（endpoint `tabletop.skill`）。LLM 白名单（`profiles.*.required_tools`）就是它的全部工具：

| tool_id | 语义 | 输入 | 输出要点 |
|---|---|---|---|
| `scene.describe` | query | `names: [str]`，可选 `camera` | 每个名字：`found`、`center_xyz`、`footprint_m`、`height_m`、`grasp_yaw_deg`、`views`、`note` |
| `scene.locate` | query | `text`、可选 `camera`、`min_score` | 同上，单物体，PyRUA `locate` 结构 |
| `plan.execute` | action | `steps: [{op: pick/place/stack_on/home, arm, …}]`、`deadline_s` | `completed_steps`、首个失败步的 `reason`、`observed` |
| `arm.pick` | action | `arm: left/right`、`target: {xyz, yaw_deg, width_m}` 或 `object: str`、`deadline_s` | `grasped`、`gripper_width_m`、`reason` |
| `arm.place` | action | `arm`、`target_xyz`、`yaw_deg`、`release_height_m`、`deadline_s` | `placed`、`final_xyz`、`reason` |
| `arm.stack_on` | action | `arm`、`base_object: str`、`deadline_s` | 先 locate 栈顶再放，返回 `top_z_after` |
| `arm.home` | action | `arm` | 回初始位，用于复位与异常恢复 |
| `arm.state` | query | `arm` | TCP 位姿、关节、夹爪开口 |
| `plan.time_left` | query | 无 | 距 180 秒上限剩余秒数、已用 LLM 轮次 |

`skill_server` 不经 Gateway 嵌套调用任何工具。它直接作为 Dora 数据面节点：
- 向 `motion_left` / `motion_right` 发 `move_pose_goal` / `move_joints_goal`，消费 `*_feedback` / `*_result`。客户端代码照抄 operator-motion 的 `motion_action_policy`（它就是 motion_server 的 Dora Action 客户端）。
- 夹爪直接发稀疏 `JointCommand`（只含 `gripper`）到驱动的 `action/gripper`，从驱动 `state` 读开口确认。不用 gripper policy 与 gripper controller。
- 从 perception 节点读物体表（同进程或 Dora 输入，builder 定）。

`arm.pick` 内部程序（PyRUA 的"代码段"）：
1. 无显式 target 时调 locate；`views < 2` 且 `note` 非空时把腕相机移到物体上方 0.25 m 重测一次。
2. 预抓取位 = 物体顶 + 0.08 m，`move_pose`，失败码 `IK_FAILED/JOINT_LIMIT_VIOLATION` → 换 yaw ± 90° 再试一次。
3. 开夹爪到 `width_m + 0.03`，直线下降到抓取高度，闭合。
4. 读夹爪开口：开口 ≤ 0.004 m 视为空抓 → 抬起、重新 locate、重试一次；仍失败返回 `reason=empty_grasp`。
5. 抬升 0.10 m，返回。

失败原因枚举：`not_found`、`unreachable`、`empty_grasp`、`dropped`、`collision_stop`、`timeout`、`driver_error`。前四项与 `timeout` 返回 `succeeded + ok:false`；`collision_stop`、`driver_error` 返回 Gateway `failed`。

## 7. Bundle 结构与 Dora 拓扑

```text
piper-dual-tabletop/
├── skill.yaml
├── SKILL.md                      # LLM 工作流 + 恢复表
├── start.sh                      # 在 $PAOS_SKILL_ROOT/../.venvs 建 uv venv、检查权重、CAN、相机、显存
├── run_scene.sh                  # 比赛入口：记场景编号/开始时间，调 paos agent -m
├── profiles/
│   ├── real-piper-dual/
│   │   ├── dataflow.yaml
│   │   ├── gateway.yaml          # 9 个 ToolSpec；image_input_ids: [fixed, wrist_left, wrist_right]
│   │   ├── piper_left.yaml       # port: can0, motion_speed_percent: 30
│   │   ├── piper_right.yaml      # port: can1
│   │   ├── motion_left.yaml / motion_right.yaml        # 各一个 group，关节名 joint1..6，不冲突
│   │   ├── controller_left.yaml / controller_right.yaml
│   │   ├── camera_fixed.yaml / camera_wrist_left.yaml / camera_wrist_right.yaml
│   │   ├── perception.yaml       # 内外参文件路径、检测器选择、世界系定义
│   │   ├── skill_server.yaml     # 两臂基座到世界系变换、高度常量、速度、中转位、工作区分界
│   │   └── recorder.yaml
│   ├── real-piper-single/        # 只起左臂链
│   └── sim-piper/                # mujoco_sim 代替驱动，单臂，复用 robot-control-sim 锁
├── nodes/
│   ├── skill_server/             # forge_tool provider + motion_server Dora Action 客户端
│   ├── perception/               # SAM3/检测 + 深度或三目定位
│   ├── piper_driver/             # adapter-robot-piper 源码快照（含 commit 记录）
│   ├── camera/                   # realsense 或 usb 适配器源码快照
│   ├── recorder/                 # 录像 + 工具日志 JSONL
│   └── run_*.sh                  # 包装脚本，激活各自 venv
└── assets/
    ├── piper_with_gripper.urdf
    └── calib/                    # 组委会相机内外参，现场填入
```

`artifacts.nodes` 锁定（registry 可解析）：`gateway`、`motion_server`、`joint_trajectory_controller`；sim profile 另加 `mujoco_sim`、`sim_control_policy`。其余节点是包内 Python 源码，dataflow 用 `${PAOS_SKILL_ROOT}/nodes/run_x.sh` 引用，`start.sh` 负责在包外建 venv（归档不允许符号链接，venv 不进包）。

`real-piper-dual` 拓扑：

```text
gateway ──tool ports──► skill_server ◄── perception ◄── camera_fixed / camera_wrist_left / camera_wrist_right
                             │                                  │ image/color → gateway.image_input_ids（证据）
                             ├─ move_pose_goal ─► motion_left ─► trajectory_goal ─► controller_left ─► joint_command ─► piper_left.action/arm
                             ├─ gripper JointCommand ───────────────────────────────────────────────────► piper_left.action/gripper
                             │        piper_left.state ─► motion_left.joint_state, controller_left.joint_state, skill_server.state_left
                             └─ 右臂同构（motion_right / controller_right / piper_right，can1）
recorder ◄── camera_fixed.image/color, skill_server.tool_log
```

驱动已核实（adapter-robot-piper `driver.py`、forge_robot README）：接受稀疏 `JointCommand`，臂只发六关节、夹爪只发 `gripper` 都可以，缺省关节沿用上次目标；runner 支持 `action/<source>` 命名空间输入。首测 `motion_speed_percent: 30`。

Gateway `tools.invoke_timeout_ms` 现有 profile 用 20000，含义未在文档确认。设为 ≥ 190000；builder 读 adapter-forge-gateway README 确认它约束的是单次握手还是整个 Action，若是后者则按最大 `deadline_s` 设置。

## 8. SKILL.md 内容要点

- 何时激活：任务文本提到桌面物体操作即激活为 primary。
- 第一步必须 `forge_task_create`，`verification.mode=audit`，criteria 写成可核对的子目标（"A 在 B 上方且 B 未移动"）。
- 工具调用顺序固定：`scene.describe` → 规划 → `plan.execute` → `scene.locate` 确认 → finalize。
- 启动 Action 用 `forge_tool_start_action(..., wait_ms=剩余时间)`，一轮拿到终态；禁止轮询 `action_status`。
- 恢复表：

| reason | 下一步 |
|---|---|
| `not_found` | 换另一台相机 `scene.locate`；仍无则换同义描述一次；再无则跳过该子目标 |
| `unreachable` | 换另一只臂；或先 `arm.place` 到中转位 |
| `empty_grasp` / `dropped` | 重新 `scene.locate` 后再 `arm.pick` 一次 |
| `collision_stop` / `driver_error` | `arm.home` 后 finalize，不再尝试 |
| `timeout` | 查 `plan.time_left`，不足 25 秒则 finalize |

- 禁止事项：不得自行发送关节值；不得在未确认 `grasped` 时执行放置。

## 9. 对 PAOS Core 与配置的改动（随整套 PAOS 提交）

1. `uv.lock`：litellm 约束 `>=1.82.1,!=1.82.7,!=1.82.8,<1.106.0` 并重锁。不修则组委会安装后所有带工具的 LLM 请求失败。
2. `forge_tool_start_action` 与 `forge_tool_action_result` 增加 `wait_ms`：在 PAOS 工具层按 `forge.pollIntervalS` 循环 GET status/result 直到终态或超时，每次 HTTP 仍受 `requestTimeoutS` 约束。超时返回 pending 与 `invocation_id`，不记 unknown。
3. `providers/effort.py`：DeepSeek 支持关思考（`reasoning_effort=none` 映射到 `thinking: disabled`）。
4. 已在本分支的 96b2151（NO_PROXY 清洗）与 511a8b8（空 `reasoning_content`）保留。
5. 配置：`agents.verification.evidenceRetention=all`、`agents.verification.timeoutS=60`、`forge.evidence.requiredImageSources=[fixed]`、`maxToolIterations=60`。

## 10. 今天的工作拆分

| 工单 | 内容 | 验收 |
|---|---|---|
| A 包骨架 | skill.yaml、三个 profile、gateway/dataflow、SKILL.md、start.sh、run_scene.sh、打包脚本 | `paos skill install --local` 通过，`sim-piper` 启动并 `GET /tools` 列出 9 个工具 |
| B skill_server | forge_tool provider（照 operator-relative-pose-policy 的 runner/lease 模式）；motion_server Dora Action 客户端（照 motion_action_policy）；夹爪 JointCommand；9 个工具 | sim-piper 上原生调用 `arm.pick/place`、`plan.execute` 成功，返回体符合 §6 |
| C perception | 三路图输入，SAM3 或替代检测器，深度优先、三目兜底，PyRUA 结构返回 | 合成场景定位误差 < 1 cm，`views`/`note` 正确 |
| D PAOS 改动 | §9 五项 | 测试通过；`paos agent` 在 sim-piper 跑完一次抓放，LLM 轮次 ≤ 7 |
| E 驱动与依赖 | piper_sdk 0.6.2、SocketCAN、相机包、SAM3 权重、各节点 venv 预建并离线缓存 wheel | `start.sh` 自检全绿（真机项标记为"明天核对"） |
| F 提交材料 | 技术说明、运行示例、执行日志、版本摘要、SHA-256 清单、LLM 申报 | 文件名与清单符合手册 |

## 11. 明天到场问评委的问题（按重要性；用户决定全部现场问）

1. **相机与标定**：三路相机有没有深度？外参相对哪个坐标系，是否包含到两臂基座的变换？若没有，明天要加手眼标定工单，会占掉大半调试窗口。
2. **运行机器与封存形式**：是自带笔记本整机封存，还是组委会从提交的 tar 包在他们的工作站上重新安装？后者意味着 `start.sh` 要现场联网建 venv、下载权重，且 GPU 没有保证。
3. **任务下发**：任务文本由裁判口述或纸面给出、队员手敲 `run_scene.sh`，还是有接口下发？物体用颜色、类别还是编号描述？决定入口脚本和检测器提示词。
4. **两臂接法**：两臂各一个 USB-CAN 直连我方主机？总线上有没有示教主臂？若已知夹爪行程与固件版本也一并给。
5. **评分依据**："闭环执行"评委看的是工具日志还是 AgentTask verdict？决定领域失败要不要进 verdict。
6. **现场生成代码**：模型在任务中现场生成的 Python 代码，在我们自己的节点里执行、只调用已声明的原语，是否允许？
7. **任务清单公布时间**：正式任务清单与成功条件是在 11 月 4 日 17:00 封存之前还是之后公布？决定能否提前写好、测好三类任务的程序。
8. **开计时前的系统状态**：裁判下达指令前，系统能否已在运行并持续观察场景（只看不动）？决定感知能否提前完成。
9. **双臂同时运动**：任务执行中两臂同时运动有没有安全限制？决定能否用双臂并行压缩动作时间。

## 12. 已做的假设

1. 现场相机为 RealSense 系列或普通 USB RGB 相机之一；perception 两条路都写，只实测其一。
2. 两臂通过 SocketCAN `can0`/`can1` 连接，Piper 固件兼容 piper_sdk 0.6.2。
3. 跑 SAM3 的机器有 ≥ 8 GB 显存；否则检测改走在线 VLM 返回 bbox。
4. LLM 用 DeepSeek-flash，关思考；申报材料写第二家 API 作为替代方式。
5. 三个场景的物体是刚性、可 top-down 抓取的桌面物件，尺寸在夹爪 0.105 m 开口内。
6. 计时在机器人停稳、裁判判定完成时结束，finalize 的验证调用不计入。

## 13. 留给 builder 决定

- `skill_server` 与 `perception` 是否同进程。
- SAM3 不可用时的检测器选择顺序。
- 中转位、安全高度、各臂工作区分界线的具体数值（放 YAML）。
- recorder 的编码参数与文件切分。
- `sim-piper` profile 是否保留 image_viewer 与 sim_control_policy。
- Gateway `invoke_timeout_ms` 的最终取值（见 §7）。
