# A 方案说明：piper-dual-tabletop（模型写代码控制单臂 PIPER）

更新：2026-11-04（感知和动作原语重构）。包名沿用 `piper-dual-tabletop`，现在只支持单臂（旧双臂 profile 在 `dev/legacy/`，代码已不兼容）。B 方案（学习策略）是另一个独立的 skill 包 `piper-vla`，两者互不引用。

## 1. 赛制要点（新版手册，落款 11-03）

- 12 题：6 道公开题（赛前训练调试），6 道保密题（正式计分，每题 100 分）。每组 3 简单 + 2 中等 + 1 困难。
- 简单"目标物品搬运"：每题给目标图像，把 1–2 件物品放进对应容器。完整进容器且稳定得 100，容器外但离边缘 ≤10 cm 得 50；每碰动一次干扰物 −10，每碰动一次容器 −50。
- 中等"物品分类整理"：积木和水果两类，1–2 件散置。指令"将桌上积木和水果分类摆放好。"；得分 = 散置件中归位的比例。
- 困难"数字逻辑摆放"：3 个数字块 + 加减乘除等号各 1，用全部数字、1 个运算符和等号排成一行等式；结构 50，成立再 50。
- 工位（照片）：桌后沿一条 PIPER，立杆顶一台深度相机朝前下方，腕部一台相机，三面白色挡板。
- 手册没写：任务如何下发、目标图像的形式、"完成信号"、执行时限。这些要现场问组委会（见 §9）。

## 2. 设计思路

1. **模型现场写代码（PyRUA-Lean 的机制）。** 只有一个工具 `robot.run(code)`。模型按“看、做、验收”分几段写：每段结束后读打印结果、看 `show` 的图，再写下一段；段内的循环、分支写在代码里；前面定义的变量和函数仍在。论文里每段平均 2.1 个动作原语，并没有“一段做完整题”的依据。
2. **任务经验写成文字（PyRUA-RSI 的思路）。** 三类题的计分要点、步骤、判断点、坑和参考代码写在 SKILL.md 的三份任务手册里。不做成包内函数：保密题与公开题同类型不同细节，函数等于对公开题过拟合；文字加参考代码让模型照着改。
3. **位置靠测量，认东西靠主模型自己看。** 位置来自 SAM3 分割加深度（`locate`，一张图、不动臂、不重试）。物体是什么、哪个编号是目标，由主模型看 `show` 附带的图判断。只有代码段内需要立刻分支时才用 `look`（另发一次无上下文的请求，关思考约 1 s）。只有组合不出来的才算原语（PyRUA 的规则）。
4. **动作按扣分规则设计。** 抓、放都是竖直下降、竖直抬起；松手只再张开 1.5 cm；放进容器时在中心、物体底部高出沿口 3 cm 处松手，夹爪不进容器；越过工作区边界的目标直接拒绝。
5. **没有任何超时停止。** 一场只在模型调用 `forge_task_finalize` 或裁判停机时结束；`time_left()` 和每次请求附带的剩余秒数只作参考。

## 3. 架构与数据流

```
run_scene.sh ──题目(+目标图像路径)──▶ PAOS agent（deepseek-flash，开思考）
                                         │ activate_skill 读 SKILL.md
                                         ▼ forge_tool_start_action("robot.run", {code}, wait=true)
Gateway :19002 ──Python 块──▶ skill_server（Dora 节点）
                                 ├ 执行器（沙箱）→ 原语 / pick·place
                                 ├ SAM3 + 深度定位 ◀── 头部相机、腕部相机（Orbbec，彩色 + 对齐深度）
                                 ├ show ──图片文件路径──▶ 工具结果 outputs.images ──▶ PAOS 主循环把图放进一条 user 消息 ──▶ 主模型
                                 ├ look ──图 + 问题──▶ DeepSeek API（无上下文）
                                 └ 台账、图片文件 → dev/scene_digest.py → 人工改手册
skill_server ──位姿──▶ motion_server（IK、轨迹）→ 轨迹控制器 → PIPER 驱动（CAN）
             ──夹爪目标──▶ 夹爪节点 gripper_stream（每 20 ms 发一次目标）→ PIPER 驱动
```

- 执行器：模型代码在受限命名空间里运行。import 只允许 numpy、math、time、itertools、functools、collections、random；不能访问文件、进程、网络、私有属性和帧对象。沙箱与 skill_server 同进程，只防误用，挡不住刻意攻击。
- 动作层（`actions.py`）：所有 IK 在 skill_server 里解好再发关节目标（`MoveJoints`），不用 motion_server 的 IK。`pick`、`place`、`drag` 动手前规划全程，够不到就不动。`tilt_deg` 固定表示向外倾斜，和 `yaw_deg` 独立（`kinematics.py` 顶部有定义）。

## 4. 给模型的接口（完整说明在 SKILL.md）

| 类别 | 调用 |
|---|---|
| 看 | `locate(text, camera=) -> list[Sight]`、`show(camera=, marks=, image=)`、`look(question, camera=, marks=, image=, json=)`、`capture(camera=)`（`rgb` + 逐像素世界坐标 `world`） |
| 拿放 | `pick(target, z=, yaw_deg=, width_m=)`、`place(target, bottom_z=, yaw_deg=)`、`drag(target, to, z=, yaw_deg=, width_m=)` |
| 动作与状态 | `move_to(xyz, yaw_deg=, tilt_deg=, path=)`、`home()`、`set_gripper(width_m, speed_m_s=)`、`state()`、`time_left()` |
| 常量 | `TABLE_Z`、`TCP_FLOOR_Z`、`MAX_OPENING_M`、`READ_UP_DEG`、`READ_RIGHT_DEG` |

三份任务手册（SKILL.md "Task playbooks"）：A 目标物进容器，B 积木水果分类，C 数字等式。每份有计分要点、步骤、坑和在假世界验证过的参考代码。

## 5. 目录

| 路径 | 内容 |
|---|---|
| `SKILL.md` | 模型读的文档：工作流、接口、三份任务手册 |
| `skill.yaml` | manifest：`real-piper`（比赛工位）、`sim-piper`（MuJoCo）；锁定的注册表节点 |
| `nodes/skill_server/src/tabletop/` | `primitives.py`（配置、命名空间、感知）、`actions.py`（动作原语）、`vision.py`（看图客户端、编号圈）、`perception.py`（相机与 SAM3）、`executor.py`（沙箱）、`dora_node.py`（节点与 Forge 工具端点）、`dora_backend.py`、`kinematics.py`、`geometry.py`、`fake.py`（假世界） |
| `nodes/skill_server/tests/` | `test_perception_primitives.py` 是感知原语的约定；`test_skill_examples.py` 原样运行三份手册的参考代码（模型看图后的判断用真值代替） |
| `profiles/real-piper/` | 驱动、运动、两台相机、`skill_server.yaml`（标定、工作区、看图配置） |
| `nodes/skill_server/src/gripper_stream.py`、`nodes/run_gripper.sh` | 夹爪节点：每 20 ms 发一次目标，闭合可以慢；每个目标的轨迹记在 `gripper_trace.jsonl` |
| `start.sh`、`run_scene.sh` | 启动钩子（venv、SAM3 权重、CAN、相机）；一场一次的入口 |
| `scripts/` | `make_bundle.sh`、`fetch_camera_nodes.sh`、`calib_compose.py`、`calibrate/`（相机标定工具和打印用标定板） |
| `dev/` | `llm_trial_booth.py`（模型照手册写代码的离线试验，还是旧的感知接口，待更新）、`scene_digest.py`（每场摘要）、`gw_run.py`（直连 Gateway）、`legacy/`（旧双臂 profile）；不进包 |

## 6. 安装与运行

```bash
./start.sh                                   # skill_server venv、SAM3 依赖与权重、CAN / 相机检查
paos skill install --local <bundle.tar.gz> -y
paos skill start piper-dual-tabletop --profile real-piper
./run_scene.sh pub-1 "<题目>" [目标图像路径]  # SCENE_LIMIT_S=<时限> 可改截止时间，默认 180 s
python3 dev/scene_digest.py                  # 本场一页摘要
```

改包后重装：同版本 `install --local` 不会覆盖，要先 `paos skill stop`、`paos skill remove piper-dual-tabletop`，再 install。

## 7. 现场要填、要确认的配置

| 项目 | 位置 |
|---|---|
| 头部、腕部相机内外参和头部深度比例（世界系 = 机械臂底座系）。11-04 已标并填好。头部相机或立杆动过就要重标（转 0.5° 桌面上约差 7 mm）；判断时看板上角点，不要看背景，挡板和箱子会动。重标：`scripts/calibrate/calibrate.py`，用法见 README | `profiles/real-piper/skill_server.yaml` |
| 两台相机序列号 | `camera_head.yaml`、`camera_wrist.yaml` |
| 固定曝光：每次启动由头部相机自动曝光测得，两台相机共用（画面太暗或过曝时调倍率，默认 1.47；`1` 即原始自动值） | 环境变量 `EXPOSURE_SCALE`（`start.sh`、`scripts/lock_exposure.py`） |
| CAN 口名（默认 `can0`）、驱动初始位不碰立杆和挡板 | `profiles/real-piper/piper.yaml` |
| 工作区边界（挡板、立杆） | `robot.workspace` |
| 裁判读等式的方向（默认裁判站在桌前，字头朝 180°，从左到右是 +y） | `robot.read_up_deg`、`read_right_deg` |
| 竖直可达高度、夹爪最大开口、holding 阈值、TCP 高度下限（指尖触桌高度 + 3 mm，桌面或夹爪变了要重测） | `robot.topdown_max_z`、`arms.*.max_opening_m`、`robot.holding_min_width_m`、`robot.tcp_floor_z_m` |
| 头部已用 1080p（`camera_head.yaml`、`K`、`find_window`）；定位仍在缩小后的 640×480 上，模型看原图 | `skill_server.yaml` |
| 关闭 PAOS evolution（避免自动改 SKILL.md） | `~/.PhyAgentOS/config.json`：`agents.evolution.enabled: false` |

## 8. 验证（11-03 晚）

- 单元测试 77 个通过，包括三份手册的参考代码在假世界里原样跑通、沙箱逃逸用例、启动死锁复现、取消与硬件故障的区分。
- PAOS 全链路（MuJoCo 单臂）：经 Gateway 跑 `robot.run`，`move_to` 误差 5.4 mm，夹爪正常。
- 真调 deepseek-flash（渲染的 8 个数字/符号块）：整图画圈读字 5/8；`inspect` 近拍转正后 8/8；`orient` 两次运行都是 8/8，约 2 s。
- 离线试验（模型照 SKILL.md 写代码，假世界，看图用标准答案代替）：搬运 4/4（约 62–84 s），分类 3/4（约 68–79 s，失败那次是沙箱误拦 numpy，已修），等式 4/4（153–173 s），含换物体、换布局、换数字的变体题。
- 11-04 真机（SDK 直连，未经驱动和 Dora）：实机 URDF `piper_with_gripper_real.urdf` 对齐固件零点，走到目标后固件位姿差 0.5 mm 以内；闭合指尖触桌 3 个点，法兰高度 147.5 mm，极差 0.44 mm；夹爪开合到 100 mm，读数 99.6 mm。
- 11-04 相机标定（`scripts/calibrate/calibrate.py`）：用竖直朝下只转 J6 的方法定板子位姿，圆心拟合 0.34–0.43 mm，两轮共 8 个位置，留一法误差不超过 2.5 mm。按运行时路径（像素 + 深度 + 外参）算板上角点，水平平均 0.6 mm、最大 1.2 mm，高度标准差 1.4 mm。头部深度比真实值短约 1%（0.8 m 处差 7 mm），已用 `depth_scale` 校正；腕部深度偏差在 ±0.5% 内（已知画面下部仍不可信），未校正。头部相机定位角点后让闭合指尖悬停 10 mm，共 12 次偏差 0.4–2.1 mm，最终外参的 4 次是 0.4–1.7 mm。以上走的是 SDK 直连和离线计算，还没经过 skill_server、motion_server。
- 大范围姿态的经典手眼标定在这台机械臂上失败（板子散开约 10 mm），J4 ≠ 0 时固件位姿不够准（J4 零点可能偏约 2°，未确证），所以手腕大幅扭转时定位会差几毫米；竖直抓取不受影响。
- 11-04 感知原语（真实头部相机 + SAM3，经 Dora 和 Gateway，不动臂）：`locate` 约 0.37 s；水果连续 8 帧的 x、y 标准差不超过 0.6 mm，顶高不超过 0.1 mm；纸箱开口连续 30 帧位置标准差约 3 mm，沿口 0.135–0.16 m；空桌面 z 中位数 −3.6 mm（标准差 1.7 mm）。腕部相机只在全链路里抓过一帧（`show("wrist")`）。
- 11-04 提示词（任务照片）：`colored number or math symbol piece` 困难题 7/7；`flat colored block` 中等题积木全中，多一个 0.43 分的误检；`fruit` 还会以约 0.5 分找到积木；`toy block`、`number`、`math symbol` 找不到。
- 11-04 主模型看图（真实 PAOS + deepseek-flash，相机专用临时 dataflow，无驱动）：`show` 的图送达主模型，它正确读出杨桃、奇异果和桃子，判定 3 号进箱，也认出 4 号是开口；错答只来自它多调的 `image` 工具（手册已写明不要用）。同一张任务照片的延迟：关思考 1.0–1.4 s，PAOS 默认 4.7–5.7 s，`reasoning_effort=max` 9–14 s。
- 11-04 驱动链首次上真机（速度 15%）：启动、停止和感知都正常。发现动作问题：`move_to(y+3cm, path="line")` 时 motion_server 的 IK 选了远处的解，J1 转 54°；1 cm 的目标落在 `final_position_tolerance_m` 内，没动却报 `reached`。动作原语重构中。
- 11-04 15:10 动作原语上真机（完整驱动链，先 15%、0.25，再 30%、0.5）：1 cm 直线误差 0.1 mm；全关节移动误差 0.4 mm，过冲 ≤0.07°，流式指令异常和 motion_server 选远解的问题都没有再出现。红心块 `pick` 成功（夹持宽度 41.6 mm）；`place` 后重测偏 6.9 mm；数字“8”`drag` 8.7 cm 偏 3.1 mm；“8”带 60° 旋转抓放偏 0.7 mm；杨桃 `place` 进纸箱成功，倾斜 70°、距底座 0.71 m、高 0.23 m 松手。默认速度耗时：`pick` 13.4 s（到上方 5.2、下降 3.2、闭合 1.8、抬起 2.9）、`place` 10.8 s、`home` 4.5 s。
- 同一轮的问题：心形块第二次被夹时被挤飞约 6 cm（无平行边）；抓杨桃时桃子的测量位置变了约 2.7 cm，很可能是全开的手指蹭到；杨桃有一次在机械臂没靠近时位置变了，原因不明。
- 11-04 17:30 RSI 第二场（把樱桃和草莓放进箱子）撞翻箱子：放完直接 `pick` 下一件，第一段是一个关节运动，从箱子上方（z≈0.28）直接到低悬停点（z≈0.055），用真实 URDF 重放，夹爪在箱子范围内最低到约 0.10 m（箱顶 0.173 m）。此前的真机路径都从初始位或悬停点出发，没走过这一段。已改成“先在两端较高的高度横移，再下降”，`pick`、`place`、`drag` 共用，重放最低 0.25 m；还没上真机复测。
- 11-04 夹爪换成 Dora 节点 `gripper_stream`（固件的夹爪指令只有位置和力矩，没有速度；Forge 控制器的配置里也没有限速项）。每 20 ms 发一次目标；`pick`、`drag` 闭合时先全速到物体外 15 mm（真机记录里实际宽度比测量值最多宽 13 mm），再以 20 mm/s 合上；手指停 250 ms 算夹住，之后继续按目标夹紧。离线验证：单元测试；真实 Dora 加模拟驱动，50.9 Hz，间隔中位数 20.0 ms，碰到物体后 0.25 s 结束；Forge 的 `mujoco_sim` 作接收端，手指实测 19.9 mm/s，稳定落后目标约 3 mm，没有误判堵转（所以按速度判定堵转，不按跟踪误差）。真机上还没跑过：平滑度、夹持力是否够、250 ms 是否够、闭合总耗时（估算约 1.1 s，原来 1.78 s）。
- 未在真机验证：腕部 `locate` 精度、夹持力上限、0.46 m 外倾斜夹扁块、`set_gripper` 之外的取消路径。

## 9. 经验积累（11-04 公开题）

1. 每跑一道公开题：`./run_scene.sh pub-<n> "<题目>" [目标图像]`，然后 `python3 dev/scene_digest.py` 出一页摘要（题目、每段代码和输出、看图问答、失败的原语、各原语耗时、发给看图模型的图）。
2. 由人按摘要改对应手册：步骤、坑、参考代码（物品的英文叫法、容器沿高度、夹取深度、摆放间距等）。
3. 跑 `tests/test_skill_examples.py` 和全部测试，`make_bundle.sh`，remove + install。
4. 17:00 封存前冻结 SKILL.md。
5. 提速待测（用户 11-04 定在 RSI 过程中跑）：夹爪 `stall_timeout_ms` 已从 1500 调到 250（要确认仍能夹紧）；`motion.line_step_m` 从 2 cm 放大（每步起停约 0.8 s，要确认直线偏差和不碰邻物）。每改一项，重跑同一道题比较总用时和成功与否。

## 10. 受本机硬件限制的参数（上 4090 工作站再放开）

| 参数 | 位置 | 本机值 | 工作站 |
|---|---|---|---|
| SAM3 精度与单帧耗时 | `skill_server.yaml` 的 `segmenter` | float16，`locate` 实测约 0.37 s | 实测后再定 |

`grep -rn HARDWARE-LIMITED .` 可找全。

## 11. 风险与待确认

- 任务下发形式、目标图像形式、完成信号、执行时限未知（明早先问组委会；问题清单见规格文档 §13.4）。
- 等式题最慢：5 块各抓放一次约 20 s，离 180 s 很近；要靠提速（驱动 `motion_speed_percent`、`motion.velocity_scale`、`approach_m`）。
- 主模型看图只测过简单题的道具；积木和数字块只用任务照片测过提示词。
- `~/.PhyAgentOS/config.json` 的 `maxTokens` 是 8192，主模型一次思考约 2.5 万字（图片消息那一轮），可能被截断；deepseek 报告里开思考配的是 131072。
- `place` 进箱改用整个箱子的 `top_z`（含翻边），不再用 `locate("empty space inside the box")`：那个词在同一个箱子上 `top_z` 在 0.06、0.13、0.17 m 之间跳。11-04 模型因此自己传了过低的 `bottom_z`，夹爪进箱把箱子推了约 11 cm，碰容器扣 50。
- 提速空间：`line_step_m` 2 cm 的起停每步约 0.8 s。
- `look` 不设超时（按"skill 包里禁止超时停止"）；网络卡住时这段代码会一直等，直到裁判停机。
- 代码审查（11-04 00:00）的未修项、以及"原语整体太冗长、防御过度，待重构"的结论，记在规格文档 `docs/superpowers/specs/2026-11-03-paizhou-track2-robot-api.md` §14。
