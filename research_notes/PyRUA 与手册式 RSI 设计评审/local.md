# 本地实现与手册式 RSI 提案核对

## 现状是否已经等同于附件提案？

### Takeaway
没有。附件是未实现的改版草案，仓库仍含一段做完整题、多视角自动测量与高层抓放函数。用户强调的“手册而非固定代码”与现有 playbook 的自我定位一致。

### Cited Findings
- SKILL 当前强制 “one block that does the whole task”，失败才回模型；但任务手册明确参考代码是起点，不是固定流程，允许按场景修改。— [SKILL 工作流](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/SKILL.md:17)、[playbook 定位](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/SKILL.md:135)
- 当前 `locate` 返回单个 Sight，`locate_all` 才是列表。`locate` 可自动移腕、多相机、重试；`locate_all` 单帧测量并按 3 cm 中心距去重。— [locate](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:454)、[locate_all](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:575)
- 当前公开 namespace 没有 capture；Frame 只有 rgb/depth/K/T_wc/meta，没有 world；Sight 没有 pixel/points。— [namespace](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:148)、[Frame](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/backend.py:110)、[Sight](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/types.py:53)
- 跨代码段共享 namespace 已实现，异常后已写入变量仍在，无事务回滚；`ok` 只表示代码没抛异常，不表示机器人任务成功。— [executor](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/executor.py:153)
- look 使用独立单条 user 消息发送图片与问题，模型名从 vision 覆盖项或 provider.defaultModel 取，默认 deepseek-flash；当前会重试 5 次，网络无 timeout。不能把附件“原语不重试”的草案当现状。— [vision config](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/vision.py:45)、[ask](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/vision.py:300)

### Inferences
- “RSI 固定程序不适合”不能直接否定手册式 RSI；应评估经验条目、示例与分支规则是否帮助模型现场生成正确动作，并保留上下文反馈闭环。
- 应保持一个稳定的公开 API 再迭代手册，避免同名 locate 在手册与运行时返回值完全不同。

### Gaps
- 未跑照片、仿真或机器人；未读取认证配置。无法验证附件的模型速度、基线与几何精度数字。

## 哪些设计合理，哪些约束应修改？

### Takeaway
可保留 locate 测量 / look 解释 / capture 计算的职责分工和阶段手册，但回模型时机应由不确定性与结果决定，不能把全题一段换成固定三段。

### Cited Findings
- 当前 look 将旧 Sight 的世界位置投影到新取的图画标记；不是同一份图中的 mask 标记。— [look](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:793)、[draw_marks](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/vision.py:112)
- 当前感知持有独立 RGB/depth 缓存；capture 只校验形状，没有按采集时间配对深度；stale RGB 可被返回，live stream 停流时等待下一帧无 timeout。— [capture](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/perception.py:135)
- holding 是夹爪闭合时宽度（可选 effort）的推断，state 直接返回上次布尔量，不能检测搬运中的滑落；held 尺寸关联最近 Sight 的 grasp_xyz。— [set_gripper](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:388)、[state](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/primitives.py:261)
- 动作返回值含 reached/reason，业务失败不抛异常；现有 pick/place 对抬起或退回的部分 move_to 结果并不检查，可能已经返回 ok=True。— [pick](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/skills.py:217)、[place_at](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/skills.py:259)、[place_in](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/skills.py:290)
- 当前 run_scene 默认 180 秒是只读 deadline 提示，不是脚本硬超时；脚本明确不设置 timeout。— [run_scene](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/run_scene.sh:5)

### Inferences
- 手册的合理默认是“观察与确认 → 对一件已确认目标完成可预测动作 → 验收/更新”，简单任务可合并正常步骤，遇到 reached=False、holding 不符、识别矛盾、目标移动就立即停在决策点回模型。print 在整段结束才回来，逐步 print 并不等于逐步反馈。
- 批量遍历旧 targets 会发生目标位置过期；每次抓放可能碰动、遮挡其他目标。可以保留完整计划，但动作前复核受影响对象和容器。
- 感知接口应带 frame_id/时间、valid-depth 比例、测量来源，支持对同一 Frame locate/look/capture，特别是 wrist 运动后。world 像素对应不是仅靠 H/W 一样就能保证。
- “身份只来自 look”过强：locate('red block') 的语义提示本就携带身份信息。合适说法是度量坐标由几何获得，歧义和任务语义由视觉问答确认。
- `home()` 是头相机遮挡时的观测策略，不能无条件每次看头相机都 home；持物状态、可达性、时间也需考虑。

### Gaps
- 未检查 PAOS 是否完全无图像回传（本地 show 明确禁用），规划图像支持应由主研究员核对核心链路。

## 附件几何示例和手册 RSI 如何落地？

### Takeaway
需把几何片段从“看起来可算”改为“前提、失败信号与补救明确的配方”。算法放在手册不等于可省略稳定的坐标/快照/状态契约。

### Cited Findings
- 当前 footprint 由可见点云最小面积外接矩形计算；top_z 是 95 分位，height = top_z - TABLE_Z，默认物体落桌；不测真实容器开口。— [footprint](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/geometry.py:192)
- 当前 place_in 在外接矩形中心放下，并以 c.size 估 opening 尺寸，实际只是外轮廓尺寸。— [place_in](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/skill_server/src/tabletop/skills.py:264)
- 原手册把数字头朝向和长轴分开：inspect 近看、orient 获取字头，之后排五块；提醒 +/×、6/9 的歧义。— [Playbook C](/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/SKILL.md:265)

### Inferences
- 附件 low = box.points[z < rim - 3 cm] 的均值不能通用地认作开口中心：mask 可能不含箱底，可能含外壁、已有物体；空集合产生 NaN，遮挡和采样不均导致偏置。需要明确箱沿/内区证据和可放区域，再给最大安全余量点；不足时换视角。
- 附件 `locate(...); f=capture(); r,c=s.pixel; world[r-10,c]-world[r+10,c]` 混用两帧；两像素可能越界、无深度、落不同物体。方向需同帧、有效邻域/平面拟合，并剔除 z 分量后算 atan2。对非平面水果，两点曲面差还可能偏离图像方向的桌面投影。
- 近方形积木长轴易跳 90°；字头方向来自印刷符号，不是外轮廓 PCA。四向读取可约束 90°/180°，不能凭外接矩形消除歧义；应允许 unknown/confidence/reobserve。
- 手册经验条目建议结构：适用前提 → 观察证据 → 可改参考片段 → 成功判据 → 异常分支 → 失败案例与实测范围。通过离线 RSI 改手册而不是不断替换固定 solve_task.py 完全合理；但属于待测的提示/记忆机制，不能直接继承论文自动迭代代码的分数。
- 验收应确认对象对应关系、容器位置/边界、完整等式可读性，不只询问泛泛“都对吗”。记录成功率、误碰数、重试/调用/总时长，比较无手册、静态手册、迭代手册；模型/API/感知和时间预算保持一致。

### Gaps
- 原论文 RSI 的确切优化产物与跨场景记忆由其他研究员核对，以上不借论文指标为手册式 RSI 背书。
