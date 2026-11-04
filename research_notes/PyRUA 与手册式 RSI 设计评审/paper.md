# PyRUA 论文证据与手册式 RSI

## 论文支持什么执行节奏？

### Takeaway
支持按反馈动态决定代码段边界；不支持强制三段，也没有否定一段完成可靠任务。手册可规定默认节奏和提前退出条件，而不是规定调用次数。

### Cited Findings
- §3.1–3.3：持久 Python 命名空间；同一 cell 内可检查结果、条件执行、局部重试；显式输出与请求图在 cell 结束返回规划模型。 [论文 §3](https://arxiv.org/html/2610.01939v2#S3)
- Appendix C：11,145 cells 上平均 2.1 个动作原语、35% 单步后看、30% NumPy 计算、84% 请求相机图。统计来自 LIBERO-PRO、RoboTwin、RoboCasa 的 700 episodes，**不是 RoboDojo**。 [Appendix C](https://arxiv.org/html/2610.01939v2#A3)
- Table 9：训练后 4B 从每次 LLM 预发中位 9 cells 变 1 cell；图像请求从每 episode 19.3 降为 0.8；仍有初始图片，不等于每步看图。18.8→44.1 是 GRPO 实验，不是手册实验。 [§5.3](https://arxiv.org/html/2610.01939v2#S5.SS3)
- §3.7 的指南消融有反例：RoboTwin 无 VLA 时，去指南让工具调用成功率 42.8→60.4，说明指南不是天然增益。 [§3.7](https://arxiv.org/html/2610.01939v2#S3.SS7)

### Inferences
- 将“先辨认—有守卫地动作—验收”用作可循环手册结构合理；任何需要重新理解身份、场景改变、状态失败处都可提前结束 cell。
- 2.1 的均值不构成上限，更不构成三段模板的因果证据；整体动作 for 循环也要按失败/观测失效停止。

### Gaps
- 没有手册式 RSI 对照实验；也未证明现有 DeepSeek 规划器能通过手册复现训练模型收益。

## 82.82% 与 51.62% 能说明手册 RSI 的收益吗？

### Takeaway
不能。数字可在同 benchmark 下描述，但两者是事先优化固定程序与在线规划的系统差异，不是“有手册/无手册”的消融。

### Cited Findings
- 官方 RSI 是每任务固定 Python 文件；评测不调 LLM；本地假 completions endpoint 第一次返回整个文件为一个 python tool call，以后返回 finish。 [RSI README](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/PyRUA_RSI/README.md), [offline_programs.py](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/PyRUA_RSI/offline_programs.py)
- 作者 RSI 汇总 seed0 2,100 episodes 平均 SR 82.82；seed0 同时是开发种子，未用于开发的 seed2 为 80.73；同 observation/API 不代表同知识与优化预算。 [RSI evaluation summary](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/evaluation/robodojo_summary_rsi.md)
- RSI README 明列预置公共任务定义、资产尺寸与 mesh、固定成功条件常量，以及仿真测出的偏移；54 文件按 task name 选择，候选程序已提前在仿真反复测试。 [RSI README](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/PyRUA_RSI/README.md)
- Lean 的 §4.2 prompt 仅 API、工具说明、当前任务 instruction 与初始相机图，无 per-task solving recipe、无跨 episode memory；§4.3 51.62 是五个能力维度等权均值，合并 episodes 是 1061/2100=50.5。 [论文 §4](https://arxiv.org/html/2610.01939v2#S4)
- pack 文件头确实标注 v20 / round9，但这是代码自带历史说明，不是论文给出的统一迭代预算，也不能简单解释成独立 20 轮实测。 [pack program](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/PyRUA_RSI/programs/pack_objects_into_box.py)

### Inferences
- 用户的 RSI 是规划模型读取的经验手册，保留在线决策；准确类比为“经验进入 prompt + 在线写 cell”，不是官方离线程序策略。
- 官方数据支持离线积累经验的潜力；不能把 31.2 个百分点归因于手册，也不能据此估算本地调用节省。

### Gaps
- 作者公告独立 RSI 技术报告 forthcoming；尚无手册/代码知识载体公平比较。

## 感知适配和真机证据应如何限定？

### Takeaway
接口原则合理；不要把不同适配器与时期的数据拼成已验证的同一系统。

### Cited Findings
- Table 15 深度版单摄 mask 内深度点直接估中心/顶部/footprint，去掉腕部复测指导；RGB 官方 RoboDojo 用多视角，无同等深度观测。 [Appendix I](https://arxiv.org/html/2610.01939v2#A9)
- world_map/NumPy 几何算例来自主对比其它 robot stacks；不能据此说 RGB RoboDojo adapter 支持 world_map。 [Appendix A](https://arxiv.org/html/2610.01939v2#A1), [Appendix C](https://arxiv.org/html/2610.01939v2#A3)
- §6 的论文比较仅仿真、粒度与真机性能未测；但当前仓库 Oct21 另有 AF8 真机演示：两次 5 块入碗，41/42 cells，各约 22 分钟，不是严谨真机 benchmark。 [论文 §6](https://arxiv.org/html/2610.01939v2#S6), [README news](https://github.com/DAGroup-PKU/PyRUA-Lean#-news)

### Inferences
- 本地 RGBD capture.world 是可取工程选择，但要保证配准与同帧/时间戳；论文不直接验证本地实现。
- 无 pick/place 是官方库的选择，论文未做其粒度消融；手册封装与薄函数都可按稳定性选择。

### Gaps
- 本地单臂相机深度、动作耗时、夹持信号、look 身份判断尚须真实链路验证；不可从双臂仿真成功率外推。
