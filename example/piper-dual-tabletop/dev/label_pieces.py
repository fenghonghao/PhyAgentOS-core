"""Label the equation pieces in a browser and build their outline templates. Cameras only, no arm.

  cd example/piper-dual-tabletop
  ~/.PhyAgentOS/skills/.venvs/piper-dual-tabletop-skill-server/bin/python dev/label_pieces.py
  open http://127.0.0.1:8765

Lay the pieces apart on the table first. Taking pictures needs the skill stopped (it holds the
camera); labelling what was already taken does not, and neither does a restart (dev/piece_capture.npz).
Labels are kept in dev/piece_labels.yaml, templates go to src/tabletop/piece_templates.json.
"""

from __future__ import annotations

import gc
import io
import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "nodes/skill_server/src"))

from tabletop import geometry as geo  # noqa: E402
from tabletop import symbols as sym  # noqa: E402
from tabletop.backend import Frame  # noqa: E402
from tabletop.perception import CameraConfig, Sam3Segmenter, reduced, window_K  # noqa: E402
from tabletop.primitives import RobotConfig  # noqa: E402

WORD = "colored number or math symbol piece"
Z = 0.008  # about the pieces' top
LABELS = PKG / "dev/piece_labels.yaml"
CAPTURE = PKG / "dev/piece_capture.npz"  # the last frames: restarts need no camera
OPERATORS = ("+", "×", "÷", "-", "=")
TEMPLATES = sym.TEMPLATE_FILE
MIN_PIXELS = 30
PORT = 8765
READ_UP_DEG = RobotConfig().read_up_deg
TILE_PX = 240

cfg = yaml.safe_load((PKG / "profiles/real-piper/skill_server.yaml").read_text())
cam_yaml = yaml.safe_load((PKG / "profiles/real-piper/camera_head.yaml").read_text())
cam = next(CameraConfig(**c) for c in cfg["cameras"] if c["name"] == "head")
K, T = window_K(cam.K, cam.find_window), cam.T()  # the 640x480 grid objects are found on
seg_cfg = dict(cfg["segmenter"])
seg_cfg.pop("kind")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
lock = threading.Lock()
state: dict = {"frames": None, "pieces": [], "verify": None, "verify_rows": [], "verify_tiles": []}


def grab(count: int) -> np.ndarray:
    import pyorbbecsdk as ob

    if subprocess.run(["pgrep", "-f", "camera_head.yaml"], capture_output=True).returncode == 0:
        raise RuntimeError("相机正被 dataflow 占用，等它停了再拍。已拍的块照常可以标注和保存。")
    exposure = int((Path.home() / ".PhyAgentOS/run/piper-dual-tabletop/camera_exposure_us").read_text())
    context = ob.Context()
    devices = context.query_devices()
    device = devices.get_device_by_serial_number(cam_yaml["device_serial"])
    pipeline = ob.Pipeline(device)
    streams = ob.Config()
    streams.enable_stream(pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                          .get_video_stream_profile(1920, 1080, ob.OBFormat.RGB, 30))
    pipeline.start(streams)
    props = ob.OBPropertyID
    device.set_bool_property(props.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
    device.set_int_property(props.OB_PROP_COLOR_EXPOSURE_INT, exposure)
    device.set_int_property(props.OB_PROP_COLOR_GAIN_INT, 16)
    for _ in range(15):
        pipeline.wait_for_frames(1000)
    frames = []
    while len(frames) < count:
        fs = pipeline.wait_for_frames(1000)
        if fs is None or fs.get_color_frame() is None:
            continue
        c = fs.get_color_frame()
        frames.append(np.frombuffer(c.get_data(), np.uint8).reshape(c.get_height(), c.get_width(), 3).copy())
        time.sleep(0.15)
    pipeline.stop()
    return np.array([reduced(Frame("head", np.asarray(cam.K, float), T, rgb), cam.find_window).rgb for rgb in frames])


def find_all(frames: np.ndarray) -> list[list]:
    """The objects in each frame. SAM3 leaves the GPU afterwards, so the idle tool takes nothing
    from a running skill."""
    import torch

    seg = Sam3Segmenter(**seg_cfg)
    found = [sym.distinct(seg.segment(rgb, WORD, key=k)) for k, rgb in enumerate(frames)]
    del seg
    gc.collect()
    torch.cuda.empty_cache()
    return found


def small_enough(mask: np.ndarray) -> np.ndarray | None:
    xy = geo.mask_to_plane(K, T, mask, Z)[:, :2]
    if len(xy) < MIN_PIXELS:
        return None
    rect = geo.min_area_rect(xy)
    return xy.mean(axis=0) if rect is not None and rect.length <= sym.MAX_PIECE_M else None


def saved_labels() -> list[dict]:
    return yaml.safe_load(LABELS.read_text(encoding="utf-8")) or [] if LABELS.exists() else []


def detect(frames: np.ndarray) -> None:
    """Each object's outline by majority of the frames. A saved label comes back only where the
    same colour still lies in the same place, so a moved piece never inherits one."""
    pieces: list[dict] = []
    for rgb, dets in zip(frames, find_all(frames)):
        for det in dets:
            c = small_enough(det.mask)
            if c is None:
                continue
            p = next((p for p in pieces if np.linalg.norm(p["c"] - c) < 0.012), None)
            if p is None:
                p = {"c": c, "votes": 0.0, "n": 0, "colors": []}
                pieces.append(p)
            p["votes"] = p["votes"] + sym.table_grid(K, T, det.mask, Z, center=p["c"])
            p["n"] += 1
            p["colors"].append(np.median(rgb[det.mask][:, :3], axis=0))
    pieces = [p for p in pieces if p["n"] >= len(frames) // 2]
    pieces.sort(key=lambda p: (round(sym.hue_of(np.median(np.array(p["colors"]), axis=0)) / 30), float(p["c"][0])))
    known = saved_labels()
    for i, p in enumerate(pieces, 1):
        p["id"] = i
        p["grid"] = p["votes"] >= p["n"] / 2
        p["color"] = np.median(np.array(p["colors"]), axis=0)
        p["axis"] = sym.long_axis_deg(p["grid"])
        p["base"] = READ_UP_DEG  # the picture shows a piece as the judges read it
        hue = sym.hue_of(p["color"])
        near = [k for k in known if np.linalg.norm(np.array(k["xy"]) - p["c"]) < 0.006
                and abs((k["hue"] - hue + 180.0) % 360.0 - 180.0) < sym.MAX_HUE_DIFF_DEG]
        p["symbol"], p["up"] = (near[0]["symbol"], near[0]["up"]) if near else ("", None)
        p["mirror"] = bool(near[0].get("mirror")) if near else False
    state.update(frames=frames, pieces=pieces)
    np.savez_compressed(CAPTURE, frames=frames)


def top_of(symbol: str, grid: np.ndarray, axis: float, up_deg: float | None) -> tuple[float, float | None]:
    """(heading of the symbol's top, turn symmetry). + × - = follow from their shape. A number's
    top is what the person turned it to, and so is ÷'s: it is almost a cross, its long side unclear."""
    if symbol in ("+", "×"):
        rows, cols = np.nonzero(grid)
        x, y = cols - cols.mean(), rows - rows.mean()
        arm = math.degrees(float(np.angle(((x + 1j * y) ** 4).sum() / (np.hypot(x, y) ** 2).sum()))) / 4.0
        return (arm if symbol == "+" else arm + 45.0), 90.0
    if symbol in ("-", "="):
        return axis + 90.0, 180.0
    # Only 8 reads the same upside down. A blobby outline can fit itself turned half a turn too, and
    # a 6 treated that way ends up as a 9.
    return float(up_deg), 180.0 if symbol in ("÷", "8") else 360.0


def mirrored(grid: np.ndarray, heading_deg: float) -> np.ndarray:
    """The outline reflected across the line through its middle along heading_deg: a piece lying on
    its back shows the mirror image of its face."""
    ux, uy = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    along = sym._X * ux + sym._Y * uy
    col = np.rint((2 * along * ux - sym._X + sym.HALF_M) / sym.CELL_M - 0.5).astype(int)
    row = np.rint((2 * along * uy - sym._Y + sym.HALF_M) / sym.CELL_M - 0.5).astype(int)
    inside = (col >= 0) & (col < sym.GRID_CELLS) & (row >= 0) & (row < sym.GRID_CELLS)
    out = np.zeros_like(grid)
    out[inside] = grid[row[inside], col[inside]]
    return out


def save(labels: dict) -> dict:
    kept, templates = [], []
    for p in state["pieces"]:
        item = labels.get(str(p["id"])) or {}
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        up = item.get("up")
        if up is None and (symbol.isdigit() or symbol == "÷"):
            return {"ok": False, "error": f"{p['id']} 号（{symbol}）还没转正"}
        if symbol.isdigit() or symbol in OPERATORS:  # anything else, a block, is only remembered as labelled
            grid = mirrored(p["grid"], p["base"]) if item.get("mirror") else p["grid"]
            top, every = top_of(symbol, grid, sym.long_axis_deg(grid), up)
            templates.append(sym.make_template(symbol, p["color"], grid, top, every))
        kept.append({"symbol": symbol, "up": None if up is None else round(float(up), 1), "mirror": bool(item.get("mirror")), "xy": [round(float(v), 3) for v in p["c"]],
                     "hue": round(float(sym.hue_of(p["color"])), 1)})
    LABELS.write_text(yaml.safe_dump(kept, allow_unicode=True, sort_keys=False), encoding="utf-8")
    TEMPLATES.write_text(json.dumps([t.to_json() for t in templates], ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return {"ok": True, "count": len(templates)}


def verify() -> None:
    """Read fresh frames with the saved templates; draw what was read on the first."""
    templates = sym.load_templates()
    frames = grab(3)
    tally: list[dict] = []
    for rgb, dets in zip(frames, find_all(frames)):
        for det in dets:
            c = small_enough(det.mask)
            if c is None:
                continue
            color = np.median(rgb[det.mask][:, :3], axis=0)
            r = sym.read_symbol(color, sym.table_grid(K, T, det.mask, Z), templates)
            item = next((t for t in tally if np.linalg.norm(t["c"] - c) < 0.012), None)
            if item is None:
                item = {"c": c, "reads": [], "overlaps": [], "top": None, "hue": sym.hue_of(color)}
                tally.append(item)
            item["reads"].append(r.symbol if r else None)
            item["overlaps"].append(r.overlap if r else 0.0)
            if r:
                item["top"] = r.top_deg
    S = 2
    image = Image.fromarray(frames[0]).resize((640 * S, 480 * S), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    rows, tiles = [], []
    for t in tally:
        uv, _ = geo.project(K, T, np.array([[t["c"][0], t["c"][1], Z]]))
        u, v = float(uv[0, 0]), float(uv[0, 1])
        reads = [x for x in t["reads"] if x]
        name = max(set(reads), key=reads.count) if reads else "?"
        color = (0, 170, 0) if reads else (220, 0, 0)
        if t["top"] is not None:
            tip = np.array([t["c"][0] + 0.03 * math.cos(math.radians(t["top"])),
                            t["c"][1] + 0.03 * math.sin(math.radians(t["top"])), Z])
            uv, _ = geo.project(K, T, tip[None, :])
            draw.line([u * S, v * S, float(uv[0, 0]) * S, float(uv[0, 1]) * S], fill=(255, 255, 0), width=3)
        draw.rectangle([u * S - 2, v * S - 10, u * S + 10 * len(name) + 8, v * S + 10], fill=(255, 255, 255))
        draw.text((u * S + 2, v * S - 9), name, fill=color, font=FONT)
        rows.append({"symbol": name, "reads": f"{len(reads)}/{len(t['reads'])}", "agree": len(set(reads)) <= 1,
                     "overlap": round(max(t["overlaps"]), 2), "hue": round(t["hue"]),
                     "xy": [round(float(v), 3) for v in t["c"]], "tile": None if t["top"] is None else len(tiles)})
        if t["top"] is not None:
            tiles.append(tile_jpeg(t["c"], t["top"], frames[0]))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=92)
    rows.sort(key=lambda r: (r["symbol"] == "?", r["symbol"]))
    state.update(verify=buf.getvalue(), verify_rows=rows, verify_tiles=tiles)


def tile_jpeg(center: np.ndarray, up_deg: float, rgb: np.ndarray) -> bytes:
    """The table around center seen from above, with heading up_deg at the top."""
    up = np.array([math.cos(math.radians(up_deg)), math.sin(math.radians(up_deg))])
    right = np.array([up[1], -up[0]])
    px, half = 120, 0.027
    s = (np.arange(px) + 0.5) / px * 2 * half - half
    cols, rows = np.meshgrid(s, s)
    xy = center + cols[..., None] * right - rows[..., None] * up
    uv, _ = geo.project(K, T, np.column_stack([xy.reshape(-1, 2), np.full(px * px, Z)]))
    c = np.clip(uv[:, 0], 0, rgb.shape[1] - 1.001)
    r = np.clip(uv[:, 1], 0, rgb.shape[0] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = (c - c0)[:, None], (r - r0)[:, None]
    tile = (rgb[r0, c0] * (1 - fc) * (1 - fr) + rgb[r0, c0 + 1] * fc * (1 - fr)
            + rgb[r0 + 1, c0] * (1 - fc) * fr + rgb[r0 + 1, c0 + 1] * fc * fr).astype(np.uint8).reshape(px, px, 3)
    buf = io.BytesIO()
    Image.fromarray(tile).resize((TILE_PX, TILE_PX), Image.BICUBIC).save(buf, "JPEG", quality=90)
    return buf.getvalue()


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>标注数字块</title>
<style>
body{font:14px sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#222;color:#fff;padding:10px 14px;display:flex;gap:10px;align-items:center;z-index:2}
button{font-size:14px;padding:6px 12px;cursor:pointer}
#msg{margin-left:8px}
.card{background:#fff;margin:6px;padding:8px;display:inline-block;border-radius:6px;vertical-align:top;text-align:center}
.card.done{outline:3px solid #2a2}
.card b{font-size:16px;margin-right:6px}
.card input{font-size:18px;width:64px;margin-right:6px}
#cards{display:flex;flex-wrap:wrap;align-items:flex-start}
.card.block{order:1;opacity:.45}
.read{display:inline-block;margin:3px;text-align:center;font-weight:bold}
.read img{width:130px;height:130px;display:block}
.mirror{font-size:12px;padding:1px 6px;margin-left:4px}
.mirror.on{background:#e22;color:#fff}
.pic{width:200px;height:200px;border-radius:50%;overflow:hidden;position:relative;cursor:grab;touch-action:none;margin-bottom:6px}
.pic img{width:100%;height:100%;display:block;user-select:none;pointer-events:none}
.pic::before,.pic::after{content:"";position:absolute;z-index:1;background:rgba(255,255,255,.55);pointer-events:none}
.pic::before{left:50%;top:0;bottom:0;width:1px}
.pic::after{top:50%;left:0;right:0;height:1px}
.hint{color:#666;font-size:12px}
#verify{padding:10px}
#verify img{max-width:100%}
td,th{padding:2px 10px;border-bottom:1px solid #ddd;text-align:left}
.bad{color:#c00;font-weight:bold}
</style>
<header>
<button onclick="post('/capture')">重新拍照识别</button>
<button onclick="saveAll()">保存并生成模板</button>
<button onclick="post('/verify')">打乱后验证</button>
<span id="msg"></span>
</header>
<div style="padding:8px 14px" class="hint">拍照前：拿走标定板和积木，块摆开、互不接触（间距约 3 cm），字头都朝相机画面上方（远离机械臂底座那一侧），这里看到的就是正的。标注：填数字或符号；数字和 ÷ 看到不正就拖正（滚轮微调）；+ × - = 不用转。字是反的（块背面朝上）点 镜像。× 可打 x，÷ 可打 /。</div>
<div id="verify"></div>
<div id="cards"></div>
<script>
const SAME={'x':'×','X':'×','*':'×','＊':'×','/':'÷','／':'÷','＋':'+','＝':'=','－':'-','−':'-','—':'-'};
function symbolOf(id){const s=document.getElementById('s'+id).value.trim();return SAME[s]||s}
let pieces=[];
function msg(t){document.getElementById('msg').textContent=t}
let mirror={};  // pieces lying on their back: the picture and the template are flipped
let angle={};  // clockwise turn of each picture; absent until the person has set it
async function load(){
  pieces=await (await fetch('/pieces.json')).json();
  angle={};mirror={};
  const box=document.getElementById('cards');box.innerHTML='';
  const stamp=Date.now();
  for(const p of pieces){
    if(p.up!==null)angle[p.id]=p.up-p.base;
    if(p.mirror)mirror[p.id]=true;
    const d=document.createElement('div');d.className=p.symbol==='积木块'?'card block':'card';d.id='c'+p.id;
    d.innerHTML=`<div class="pic" onpointerdown="drag(event,${p.id})" onwheel="spin(event,${p.id})"><img id="i${p.id}" src="/tile/${p.id}.jpg?${stamp}" draggable="false"></div>
    <div><b>${p.id}</b><input id="s${p.id}" value="${p.symbol||''}" oninput="mark(${p.id})"><span class="hint">色相 ${p.hue}</span><button id="m${p.id}" class="${p.mirror?'mirror on':'mirror'}" onclick="flip(${p.id})">镜像</button></div>`;
    box.appendChild(d);
    turned(p.id);
  }
  msg(pieces.length+' 块');
}
function turned(id){document.getElementById('i'+id).style.transform=`rotate(${angle[id]||0}deg) scaleX(${mirror[id]?-1:1})`;mark(id)}
function flip(id){mirror[id]=!mirror[id];document.getElementById('m'+id).classList.toggle('on',mirror[id]);turned(id)}
function drag(ev,id){
  const r=ev.currentTarget.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2;
  const at=e=>Math.atan2(e.clientX-cx,cy-e.clientY)*180/Math.PI;
  const from=at(ev),start=angle[id]||0;
  const move=e=>{angle[id]=start+at(e)-from;turned(id)};
  const up=()=>{removeEventListener('pointermove',move);removeEventListener('pointerup',up);
    if(angle[id]===undefined)angle[id]=start;turned(id);document.getElementById('s'+id).focus()};
  addEventListener('pointermove',move);addEventListener('pointerup',up);
  ev.preventDefault();
}
function spin(ev,id){ev.preventDefault();angle[id]=(angle[id]||0)+(ev.deltaY>0?2:-2);turned(id)}
function mark(id){const s=symbolOf(id);document.getElementById('c'+id).classList.toggle('done',!!s)}
async function saveAll(){
  const labels={};for(const p of pieces){labels[p.id]={symbol:symbolOf(p.id),up:p.base+(angle[p.id]||0),mirror:!!mirror[p.id]}}
  msg('保存中…');
  const r=await (await fetch('/save',{method:'POST',body:JSON.stringify(labels)})).json();
  msg(r.ok?`已生成 ${r.count} 个模板，可以打乱后验证`:r.error);
}
async function post(path){
  if(path=='/capture'&&!confirm('重新拍照后，没保存的标注会丢。继续？'))return;
  msg('处理中，约 10 秒…');
  const r=await (await fetch(path,{method:'POST'})).json();
  if(!r.ok){msg(r.error);return}
  if(path=='/capture'){document.getElementById('verify').innerHTML='';await load()}
  else{
    const rows=r.rows.map(x=>`<tr class="${x.symbol=='?'||!x.agree?'bad':''}"><td>${x.symbol}</td><td>${x.reads}</td><td>${x.agree?'':'不一致'}</td><td>${x.overlap}</td><td>${x.hue}</td><td>${x.xy}</td></tr>`).join('');
    const stamp=Date.now();
    const tiles=r.rows.filter(x=>x.tile!==null).sort((a,b)=>(parseInt(a.symbol)||99)-(parseInt(b.symbol)||99)).map(x=>`<div class="read">${x.symbol}<img src="/read/${x.tile}.jpg?${stamp}"></div>`).join('');
    document.getElementById('verify').innerHTML=`<div class="hint">读到的块，按读到的字头转正：字应该都是正的。</div>${tiles}<br><img src="/verify.jpg?${stamp}"><table><tr><th>读成</th><th>帧数</th><th></th><th>重合度</th><th>色相</th><th>位置</th></tr>${rows}</table>`;
    msg('验证完成：黄线指向字头。红色 ? 是没读：积木，或外形太接近、留给模型看图的块');window.scrollTo(0,0);
  }
}
load();
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, body: bytes, kind: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value) -> None:
        self._send(json.dumps(value, ensure_ascii=False).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/pieces.json":
            self._json([{"id": p["id"], "symbol": p["symbol"], "up": p["up"], "mirror": p["mirror"], "base": p["base"],
                         "hue": round(sym.hue_of(p["color"]))} for p in state["pieces"]])
        elif path.startswith("/tile/"):
            p = state["pieces"][int(path.split("/")[2].split(".")[0]) - 1]
            self._send(tile_jpeg(p["c"], p["base"], state["frames"][0]), "image/jpeg")
        elif path.startswith("/read/"):
            self._send(state["verify_tiles"][int(path.split("/")[2].split(".")[0])], "image/jpeg")
        elif path == "/verify.jpg" and state["verify"]:
            self._send(state["verify"], "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        with lock:
            try:
                if self.path == "/capture":
                    detect(grab(8))
                    self._json({"ok": True})
                elif self.path == "/save":
                    self._json(save(json.loads(body)))
                elif self.path == "/verify":
                    verify()
                    self._json({"ok": True, "rows": state["verify_rows"]})
                else:
                    self.send_error(404)
            except Exception as exc:  # shown in the page: this is a bench tool
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    detect(np.load(CAPTURE)["frames"] if CAPTURE.exists() else grab(8))
    print(f"{len(state['pieces'])} pieces. Open http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
