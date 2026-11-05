# 11-05 早上 8:00–10:00 测试与部署

机械臂上一次运行是 11-04 20:02。下面的改动都是之后写的，只过了离线测试（377 个，11-05 00:0x 全过），没有上过真机。

分数：简单 3 题 + 中等 2 题 = 500 分，困难 1 题 = 100 分。所以先保夹爪和简单、中等，再做等式。

## 一、没上过真机的改动

| # | 改动 | 影响 | 离线证据 | 测试 |
|---|---|---|---|---|
| 1 | 夹爪换成自写节点 `gripper_stream`，替换 Forge 控制器 | **所有题** | 真实 Dora + 模拟驱动 50.9 Hz；mujoco_sim 19.9 mm/s | T1 |
| 2 | `pick`、`drag` 慢闭合：快到物体外 15 mm，再 20 mm/s；停 250 ms 算夹住（原 1500 ms） | **所有题** | 假世界 | T1 |
| 3 | `set_gripper(speed_m_s)` 手动慢闭合 | 模型可选用 | 假世界 | T1 |
| 4 | 桌面放置间隙 1 cm → 2 mm | 中等、困难；有撞桌风险 | 假世界 | T2 |
| 5 | 手册 A、B 小改（含“手指别对准两颗中间”，19:48 加入后没重跑） | 简单、中等 | 无 | T3 |
| 6 | 两轮匹配和新模板（门槛放宽到 0.70，加 1080p 带颜色外形） | 困难 | 3 个布局留一验证：读出 91%，0 错。门槛就是在这 3 个布局上挑的，没走过真实 `locate` | T4 |
| 7 | 腕部 `locate` 读数字（Block 1，外倾 25°，z = 0.08） | 困难 | 无；外圈可达性也没实测 | T4 |
| 8 | `look` 同色候选读数字 | 困难 | 72 张真实放大图 54% | T4 |
| 9 | 查阅图 `PIECE_CHART`、`PIECE_IMAGES`；4 和 × 的图待补拍 | 困难 | 假世界 | T4 |
| 10 | 读字方向改为 `READ_UP_DEG=0`、`READ_RIGHT_DEG=-90` | 困难：方向错，结构 50 分可能全丢 | 无（用户按官方示意图定） | T5 |
| 11 | 手册 C 重写：模型选位、一次摆完、留腕部转角、大转角两次抓、可挪无关块 | 困难 | DeepSeek 选位 9 次，每次 19–36 s；从没在真机跑通（上次 176 s 摆 2/5） | T5 |
| 12 | 监看页聊天式自动贴底 | 只影响监看 | 页面加载过 | 顺带看 |
| 18 | 一段下降：`line_step_m` 0.02 → 0.10，`line_joint_jump_rad` 0.35 → 0.6。只在已安装的副本里试，通过才写回仓库 | **所有题**：抓、放、拖的直线段 | 真实 URDF 逆解：25 个 8 cm 下降都是 1 个目标，都能规划，中途横向最大 3.5 mm | T0b |

环境和部署，也都没跑过：

| # | 项目 | 测试 |
|---|---|---|
| 13 | 共用电脑安装：`HOME` 隔离、路径改写、PAOS 跑在 Python 3.12 | D1 |
| 14 | 4090 上的 torch cu128 和 SAM3 耗时 | D2 |
| 15 | 现场网络到 DeepSeek 的时延 | D2 |
| 16 | Dora 共享内存补丁长时间运行（此前最长只跑过短测） | 全程监视 |
| 17 | 监看页走 SSH 隧道 | D2 |

明早不测：分段视觉对准（还没实现）、B 方案。头部相机 USB 掉线是已知未解决的问题，只做监视。

## 二、时间表

假设：测试时机械臂和相机先接笔记本，同时有人往共用电脑部署，最后切过去。

| 时间 | 测试线（机械臂旁） | 部署线（共用电脑） | 决策点 |
|---|---|---|---|
| 08:00–08:15 | T0 安装、起链路、复核标定 | D0 问组委会，传底座包 | |
| 08:15–08:20 | T0b 一段下降 | D1 解压、改路径、无硬件自检 | **08:20 定下降配置** |
| 08:20–08:35 | T1 夹爪 | D1 继续 | **08:35 夹爪去留** |
| 08:35–08:45 | T2 贴桌放置 | D2 测时、测网络、仿真全流程 | **08:45 定间隙** |
| 08:45–09:05 | T3 简单、中等回归 | D2 继续 | **09:05 这两类过了才往下走** |
| 09:05–09:20 | T4 读字 | D3 同步测试线定下的改动 | |
| 09:20–09:40 | T5 困难端到端 2 场 | D4 备好 udev、CAN、video 组 | **09:40 冻结代码** |
| 09:40–10:00 | T6 硬件切到共用电脑，最终验收（两条线合并） | | |

如果机械臂只能接共用电脑：T0 等 D1 过了再开始，约 08:30 起；T1–T3 不变，T4 去掉 `look`，T5 只跑 1 场，T6 省掉切硬件，只做验收。

**晚了按这个顺序砍：** 补拍 → 第二场困难 → `look` → T4 其余部分。夹爪、简单和中等回归、T6 不砍。

全程开着（对应 #16）：

```bash
watch -n 30 'df -h /dev/shm | tail -1; free -h | sed -n 2,3p; journalctl -k --since "08:00" | grep -c "USB disconnect"'
```

`/dev/shm` 持续上涨或 USB 计数变化就告诉大家。不要删 `/dev/shm` 里别人的文件。

## 三、测试步骤

以下代码块存成文件后用这条命令运行（同一条链路内变量会保留）：

```bash
cd /home/bsb/PhyAgentOS-core/example/piper-dual-tabletop
/home/bsb/PhyAgentOS-core/.venv/bin/python dev/gw_run.py 19002 - < block.py
```

这些代码块在假世界里跑过，只确认了名字和参数没错（`look` 假世界没配，没跑到），不代表真机行为。

**每次动臂前：有人手放在急停上，桌上没有手。今天人多，远程操作前要先和现场确认。**

### T0 安装和起链路（08:00–08:15）

1. **接 CAN 前先停旧链路。** 11-04 19:30 的 real-piper 数据流还在 `Running`，别让它在 CAN 恢复时带着旧状态连上：`paos skill stop piper-dual-tabletop`。
2. 看内存：`df -h /dev/shm; free -h`。
3. 上电，第一次上电先查夹爪线（反馈全 0 就是没插）。拉起 can0，确认 1 Mbit/s。
4. 确认读出率那边的验证结束后，打包并安装：
   ```bash
   scripts/make_bundle.sh
   paos skill install --local /home/bsb/PhyAgentOS-core/dist/piper-dual-tabletop-0.1.0.tar.gz -y
   grep run_gripper ~/.PhyAgentOS/skills/piper-dual-tabletop/profiles/real-piper/dataflow.yaml
   ```
   `grep` 必须有输出，否则就是被跳过了。
5. 把下降配置改成 T0b 要测的值（已经过了 08:10 就不改，跳过 T0b）。运行时读的是已安装的副本，改仓库里的那份不生效。本文所有现场调参都一样：改 `~/.PhyAgentOS/skills/piper-dual-tabletop/` 里的副本，再重启链路；定下来后写回仓库。
   ```bash
   F=~/.PhyAgentOS/skills/piper-dual-tabletop/profiles/real-piper/skill_server.yaml
   sed -i 's/^  line_step_m: 0.02$/  line_step_m: 0.10\n  line_joint_jump_rad: 0.6/' "$F"
   grep -n "line_" "$F"
   ```
6. 复核标定（链路启动前做，它直连 can0）：桌面放标定板，运行 `calibrate.py point cal4`（venv：`~/.PhyAgentOS/skills/.venvs/piper-dual-tabletop-calibrate`）。指尖悬停在角点上方 10 mm，偏差 ≤ 2 mm 算过。超过就重标头部：`collect --head-only`，约 4 分钟。判断相机动没动要看板上角点，不看背景。
7. `paos skill start piper-dual-tabletop -p real-piper`，打开 http://127.0.0.1:8780 看画面。
8. 确认画面是活的：

   ```python
   import time
   a = capture("head").rgb
   time.sleep(0.5)
   b = capture("head").rgb
   print("changed", round(float((a != b).mean()), 4))
   ```

   结果是 0 就是相机冻结，重启链路并查 USB。

### T0b 一段下降（08:15–08:20，对应 #18）

目的是去掉下降时的走走停停（原来 8 cm 分 4 段，约 3.2 s）。最先测，这样后面的测试都跑在定下来的配置上。这也是新夹爪节点第一次上真机：`set_gripper(0.06)` 就报错的话，直接跳到 T1 的 08:35 决定。

桌面清空。下降终点比最低高度高 1 cm，不贴桌：

```python
import time
set_gripper(0.06)
for x, y in ([0.24, 0.12], [0.30, 0.28], [0.22, 0.05], [0.40, 0.10], [0.34, 0.25]):
    move_to([x, y, TCP_FLOOR_Z + 0.09], tilt_deg=0)
    t = time.time()
    r = move_to([x, y, TCP_FLOOR_Z + 0.01], tilt_deg=0, path="line")
    print(x, y, "down", round(time.time() - t, 2), "s", r.reached, r.reason, r.err_m)
    move_to([x, y, TCP_FLOOR_Z + 0.09], tilt_deg=0, path="line")
home()
```

离线用真实 URDF 逆解跑过这段代码：都能到，总关节目标数从 50 个降到 14 个。

通过：下降连续不停顿、没有跳变，每段不超过 2 s，`err_m` 不超过 0.0013。

- **通过：** 保留新配置，并写回仓库（打包和共用电脑都从仓库来）：
  ```bash
  cd /home/bsb/PhyAgentOS-core/example/piper-dual-tabletop
  sed -i 's/^  line_step_m: 0.02$/  line_step_m: 0.10\n  line_joint_jump_rad: 0.6/' profiles/real-piper/skill_server.yaml
  ```
- **不通过：** 改回去，重启链路，继续 T1：
  ```bash
  F=~/.PhyAgentOS/skills/piper-dual-tabletop/profiles/real-piper/skill_server.yaml
  sed -i -e 's/^  line_step_m: 0.10$/  line_step_m: 0.02/' -e '/^  line_joint_jump_rad: 0.6$/d' "$F"
  paos skill stop piper-dual-tabletop
  paos skill start piper-dual-tabletop -p real-piper
  ```

**后面如果手指蹭到物体、邻近物体被碰动，先改回这两行，再判断是不是夹爪的问题。**

### T1 夹爪（08:20–08:35，对应 #1 #2 #3）

旧版写法用 `bottom_z=0.01` 放回，这样这一步不受 2 mm 间隙影响。

**a. 空载**

```python
print(set_gripper(0.06))
print(set_gripper(0.0, speed_m_s=0.02))
print(set_gripper(0.06))
```

看轨迹：

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".PhyAgentOS/logs/piper-dual-tabletop/gripper_trace.jsonl"
for line in p.read_text().splitlines()[-3:]:
    t = json.loads(line); s = t["samples"]
    v = abs(s[-1][2] - s[0][2]) / max(s[-1][0] - s[0][0], 1e-3) * 1000 if len(s) > 1 else 0
    print(t["goal"], t["max_velocity"], t["error_code"], "stalled", t["stalled"], "reached", t["reached_goal"],
          "final", round(t["final"], 4), "elapsed", round(t["elapsed_s"], 2), "avg mm/s", round(v, 1))
PY
```

通过：慢闭合约 20 mm/s、曲线平滑、`reached` 为真、没有误判 `stalled`；机械臂本身不动。

**b. 抓取**：桌上每次只放一件。依次测积木、数字块、猕猴桃、草莓、樱桃，各 2 次，共约 10 次。

```python
TEXT = "kiwi"   # 依次换成要测的物体
home()
s = locate(TEXT)[0]
print(np.round(s.center_xyz, 3).tolist(), np.round(s.size, 3).tolist())
r = pick(s, z=TCP_FLOOR_Z if s.grasp_xyz is None else None)
print("pick", r.ok, r.reason, r.width_m, "holding", state().holding)
```

看几秒，确认抬起没滑落，再放回：

```python
p = place([round(float(v), 3) for v in s.center_xyz[:2]], bottom_z=0.01)
print("back", p.ok, p.reason)
home()
```

**c. 拖动**，一次：

```python
home()
s = locate("colored number or math symbol piece")[0]
d = drag(s, [s.center_xyz[0], s.center_xyz[1] + 0.05], z=TCP_FLOOR_Z if s.grasp_xyz is None else None)
print("drag", d.ok, d.reason, d.width_m)
home()
```

通过：`holding` 判断全对、不滑落、物体没被弹走。闭合约 1.1 s（估算，原来 1.78 s）。

**08:35 决定：**

- 通过：继续。
- 夹不紧或误判：调 `gripper_controller.yaml` 的 `stall_timeout_ms`、`actions.py` 的 `CLOSE_SPEED_M_S`，最多调到 08:45。
- 还不行，恢复旧方案（用户确认后执行），然后重新打包安装，再跑一遍 b：
  ```bash
  cd /home/bsb/PhyAgentOS-core
  git diff --binary backup/piper-gripper-before-stream backup/piper-gripper-stream-done -- example/piper-dual-tabletop | git apply -R
  ```
  只撤销这次夹爪改动，不碰其他会话的工作。

### T2 贴桌放置（08:35–08:45，对应 #4）

桌上只放一块数字块，`SLOT` 依次换成近、中、远三处，各跑一次：

```python
SLOT = [0.24, 0.12]   # 再换 [0.30, 0.28]、[0.38, 0.42]
home()
s = locate("colored number or math symbol piece")[0]
r = pick(s, z=TCP_FLOOR_Z if s.grasp_xyz is None else None)
print("pick", r.ok, r.reason, r.width_m)
if r.ok:
    p = place(SLOT)
    print("place", p.ok, p.reason, p.release_xyz, p.tilt_deg)
home()
after = min(locate("colored number or math symbol piece"), key=lambda w: np.linalg.norm(np.array(w.center_xyz[:2]) - SLOT))
print("offset mm", np.round((np.array(after.center_xyz[:2]) - SLOT) * 1000, 1).tolist())
show("head", marks=[after])
```

再放一个水果测一次。

通过：不碰桌（听声音、看手指）、不弹跳、驱动不报错、落点偏差 ≤ 5 mm。

不通过：把 `profiles/real-piper/skill_server.yaml` 的 `place_gap_m` 改成 0.005，重启链路再测一处。

### T3 简单、中等回归（08:45–09:05，对应 #1 #2 #4 #5）

各跑 1 场，每场约 3 分钟加摆场。场景按 `任务示例/` 的初态图摆。

```bash
./run_scene.sh simple-1105 "把樱桃和草莓放进箱子里"
./run_scene.sh medium-1105 "将桌上积木和水果分类摆放好。"
python3 dev/scene_digest.py --last 2
```

通过：目测完全符合评分规则；没碰动容器（每次 −50）；180 s 内。上次参考：简单 163.9 s，中等 155.7 s。

不通过：先判断是模型的问题还是原语的问题。原语退化（比夹爪之前差）就回到 T1 的决定；**09:05 前这两类没过，T4、T5 让路，时间继续花在这里。**

### T4 读字（09:05–09:20，对应 #6 #7 #8 #9）

**a. 头部**：25 块全部摆上桌，用一个新布局（不要用建模板时的 3 个）：

```python
home()
pieces = locate("colored number or math symbol piece")
for i, s in enumerate(pieces, 1):
    print(i, s.symbol, s.top_deg, s.top_every_deg, s.color, np.round(s.center_xyz[:2], 3).tolist())
print("read", sum(s.symbol is not None for s in pieces), "of", len(pieces))
show("head", marks=pieces)
show(image=PIECE_CHART)
show(image=PIECE_IMAGES["6"])
```

人工对照真值，记下读出数、读错数、方向错的块。**通过：0 错。读出率其次**，离线是 91%。

有错读：把 `symbols.py` 顶部对应门槛提到错读之上，只在 09:20 前改。

**b. 腕部和 `look`**：`LOOK` 填头部没读出的块的编号，以及同色候选（表在手册 Playbook C）。优先 13、16、6、9、=，约 3 分钟。

```python
LOOK = {1: ["÷", "6", "8", "13", "14", "16"]}   # 编号 -> 同色候选
for i, same_colour in LOOK.items():
    s = pieces[i - 1]
    h = math.atan2(s.center_xyz[1], s.center_xyz[0])
    r = move_to([s.center_xyz[0] - 0.04 * math.cos(h), s.center_xyz[1] - 0.04 * math.sin(h), 0.08],
                yaw_deg=math.degrees(h) + 90, tilt_deg=25, path="joint")
    print(i, "reached", r.reached, r.reason)
    if not r.reached:
        continue
    near = locate("colored number or math symbol piece", camera="wrist")
    print(i, "wrist", [(w.symbol, w.top_deg) for w in near])
    print(i, "look", look(f"A flat plastic piece lies in the middle of this image, cut in the shape of a number or a sign, "
                          f"turned any way. It is one of: {', '.join(same_colour)}. Which one, and which side of the image "
                          f"does its top point to: up, down, left or right?", camera="wrist"))
    show("wrist")
home()
```

记下腕部和 `look` 各读对几块、每块耗时。腕部有错读，手册 C 就去掉这一级，直接用 `look`。

**c. 外圈可达性**（假世界答不了这个问题，一定要真机测）：

```python
for xy in ([0.22, 0.08], [0.34, 0.32], [0.38, 0.44], [0.38, 0.08], [0.45, 0.39]):
    h = math.atan2(xy[1], xy[0])
    r = move_to([xy[0] - 0.04 * math.cos(h), xy[1] - 0.04 * math.sin(h), 0.08],
                yaw_deg=math.degrees(h) + 90, tilt_deg=25, path="joint")
    print(xy, "reached", r.reached, r.reason, r.err_m)
home()
```

够不到的范围写进手册 C，让模型把这些块留给头部读。

**d. 补拍**（有时间才做，约 3 分钟）：4 和 ×，步骤见 `dev/piece_chart_reshoot.md`。

### T5 困难端到端（09:20–09:40，对应 #10 #11）

按 `任务示例/困难_末态示意图.jpg` 摆：3 个数字块，加减乘除等号各 1 块。

```bash
./run_scene.sh hard-1105-1 "用桌上的数字和符号块摆出一个正确的等式。"
python3 dev/scene_digest.py --last 1
```

记录：开始移动的时间（目标 ≤ 60 s）、摆完几块、总耗时、模型选的位置是否合理、大转角两次抓是否触发、算式对不对。

**先看第一场的方向：** 从评委一侧看，字头和行方向是否和示意图一致。不一致就改 `primitives.py` 的 `read_up_deg`、`read_right_deg`，这是立刻能修的事。

第二场换一组数字和摆放。

通过（按得分）：5 块排成合法结构（50）、等式成立（50）、180 s 内。

### T6 切到共用电脑做最终验收（09:40–10:00）

09:40 冻结代码。此后只修导致不能运行的问题；没验证的改动一律不进。

1. 推送冻结版本到共用电脑，记下 git 版本。提交由用户决定。
2. `source env.sh`，确认 `paos`、`dora` 都来自我们的目录，Dora 是 0.4.1。
3. 切 CAN 和两台相机的 USB。查 `df -h /dev/shm`、`nvidia-smi`，确认 Dora coordinator 是我们自己的。不是就先问，不要 `dora destroy`。
4. 起链路，确认画面是活的。相机和立杆没动就不用重标。
5. 简单 1 场，有时间再跑中等或困难 1 场。
6. 拉回日志。之后按组委会的要求封存。

## 四、部署步骤

需要今晚先打好底座包。详细设计见 11-04 夜的讨论，脚本还没写。

- **D0（08:00）问组委会：** 能否给单独的 Linux 账号；系统版本（Dora CLI 要 glibc ≥ 2.34，Ubuntu 20.04 不行）；怎么分时；“封存”具体怎么做；相机和机械臂是不是同一套（换了要改序列号并重标）。
- **D1（08:15）安装和自检：** 解压，改写路径，`source env.sh`；确认 `paos --version`、`dora --version`、skill venv 能 `import dora, torch` 且 CUDA 可用、`/dev/shm` 和端口 6012、19002、8780 的占用情况。
- **D2（08:35）测试：**
  - SAM3 单帧耗时，只记录。**明早不要改精度**：读字门槛是在 float16 的 mask 上挑的，换精度等于作废这些验证。
  - DeepSeek 单次调用时延。参考：关思考约 2.0 s，开思考约 9.4 s。
  - 没有别人占用 Dora 时跑一遍 `sim-piper` 全流程。
  - `ssh -L 18780:127.0.0.1:8780 booth`，在本机打开 http://127.0.0.1:18780。
- **D3（09:05）同步：** 测试线定下的改动按“笔记本 → 共用电脑”单向推送，现场不直接改代码。
- **D4（09:20）备硬件：** udev 规则、`video` 组、`can-utils`、can0 激活方法，需要 sudo 的提前和组委会说好。

`config.json` 要从本机复制后只留 deepseek 这一个 key，保留 `maxTokens: 16384`：这个值不在仓库里，丢了模型的长回复又会被截断。

## 五、记录

每项测试记一行结果，放在这个文件最后。没跑的写“没跑”，不要留空。

| 时间 | 测试 | 结果 | 改了什么 |
|---|---|---|---|
| 09:15 | D1 共用电脑 exai@192.168.4.186 安装 | 通过：/home/exai/bsb-paos（`HOME` 隔离，13 GB），effde75，无硬件预检全过，节点校验通过 | 目标机新建该目录；密钥没传，`apiKey` 留空 |
| 09:17 | D2 sim-piper 全流程 | 通过：关节、直线、夹爪、回 home 都跑通，`/dev/shm` 没涨 | 无 |
| 09:20 | D2 SAM3（RTX 4090） | 加载 3.4 s；换帧约 0.08 s/次；峰值 2171 MiB；float16 未改 | 无 |
| 09:13 | D0 相机是否同一套 | 不是：头部 AY6Z36300A4（USB3），腕部 AY6V16300WZ（USB2），外参作废 | 配置里的序列号没改，预检会失败、不会启动 |
| 09:19 | 目标机相机出流 | 头部开深度就 USB 断开（2 种分辨率都复现），只开彩色正常；腕部彩色 + 深度约 31 fps | 无，需现场查线缆/端口 |
| | D2 DeepSeek 时延 | 没跑：目标机没有密钥（接口可达，无密钥请求 0.55 s 返回 401） | |
| | D2 监看页 SSH 隧道 | 没跑：预检没过，监看页未启动 | |
| | D4 can0 | 没跑：需要 sudo 密码，当前 DOWN | |
| | real-piper 链路 | 没跑：等 can0、相机和标定 | |
