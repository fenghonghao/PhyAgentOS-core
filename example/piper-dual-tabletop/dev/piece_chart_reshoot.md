# 查阅图补拍（比赛当天早上）

查阅图 `nodes/skill_server/src/tabletop/piece_chart.jpg` 里还有几块不够好，有真机时补拍。缺了也能比赛，这张图只是给模型对照用的。

| 块 | 现状 | 优先级 |
|---|---|---|
| 4 | 3 个布局里旁边都挨着别的块，角上露出邻块 | 要拍 |
| × | 从比赛前一场的放大图裁的，比别的块模糊 | 要拍 |
| 16、10 | 边缘还留一点点别的颜色 | 有时间再拍 |

## 怎么拍

全程约 3 分钟，机械臂不动，只用头部相机。

1. 确认链路在运行，手臂在初始位，手离开桌面后，头部画面在动（监看页 http://127.0.0.1:8780/）。
2. 把要拍的块平放在桌面中部（x 约 0.25–0.40 m，y 约 0.10–0.35 m），**每块周围 10 cm 内没有别的东西**。块要正面朝上。
3. 手离开桌面，运行下面的代码块（把它存成文件，用 `dev/gw_run.py 19002 -` 从标准输入运行）：

   ```python
   f = capture("head")
   for s in locate("colored number or math symbol piece"):
       r, c = s.pixel
       print(s.symbol, s.top_deg, s.top_every_deg, [round(v, 3) for v in s.center_xyz[:2]], s.pixel)
       show(image=f.rgb[r - 48:r + 48, c - 48:c + 48].repeat(3, axis=0).repeat(3, axis=1))
   ```

4. 图片按顺序存在 `~/.PhyAgentOS/logs/piper-dual-tabletop/show/`（`*_image.jpg`），打印的每一行对应一张。把用得上的复制到 `dev/piece_layouts/`。

## 拍完怎么换

把图片和这一行打印的内容（块、`top_deg`、位置）交给我，我换进 `dev/build_piece_chart.py` 重新生成，确认后重新打包安装。× 的 `times_from_a_tile()` 就是现成的例子。

`top_deg` 是 `None` 时，记下字头在图里朝哪边，我按这个转正。
