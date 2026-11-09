"""Generate a single-file, zero-install annotation page.

The 18 task images are inlined as base64, so the page works by double-clicking it — no server,
no account, no file picker. Boxes live in localStorage and export as one JSON.

    python scripts/build_annotator.py

Then: open annotation_task/annotate.html, drag boxes (keys 1-6 pick the class),
click 导出 and send the downloaded JSON back.
"""
import base64
import json
from pathlib import Path

TASK = Path("annotation_task")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>标注 · NEU-DET 18 张</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#14171c; color:#e6e8eb;
         font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:12px 18px;
           background:#1c2128; border-bottom:1px solid #2b3138; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.3px; }
  .progress { margin-left:auto; font-size:13px; color:#8b98a8; font-variant-numeric:tabular-nums; }
  main { display:flex; gap:20px; padding:18px; align-items:flex-start; }
  #stage { position:relative; background:#0d1013; border:1px solid #2b3138; border-radius:8px;
           padding:8px; }
  canvas { display:block; cursor:crosshair; border-radius:4px; }
  aside { width:230px; }
  .cls { display:flex; align-items:center; gap:8px; padding:7px 10px; margin-bottom:6px;
         border:1px solid #2b3138; border-radius:6px; cursor:pointer; font-size:13px; }
  .cls .key { width:18px; height:18px; border-radius:4px; display:grid; place-items:center;
              font-size:11px; color:#0d1013; font-weight:700; }
  .cls.on { border-color:#5b9cf6; background:#1b2530; }
  .cls .n { margin-left:auto; color:#8b98a8; font-size:12px; }
  .row { display:flex; gap:8px; margin-top:12px; }
  button { flex:1; padding:9px 10px; border-radius:6px; border:1px solid #2b3138;
           background:#232a33; color:#e6e8eb; font-size:13px; cursor:pointer; }
  button:hover { background:#2b333d; }
  button.primary { background:#2f6fd0; border-color:#2f6fd0; }
  button.primary:hover { background:#3a7ee0; }
  .hint { margin-top:14px; font-size:12px; line-height:1.8; color:#8b98a8; }
  .hint b { color:#c9d2dd; }
  .done { outline:2px solid #2ca02c; }
</style>
</head>
<body>
<header>
  <h1>手工标注 · NEU-DET（18 张）</h1>
  <span style="font-size:12px;color:#8b98a8">拖拽画框 → 按键选类别 → 回车下一张</span>
  <span class="progress" id="prog">1 / 18</span>
</header>
<main>
  <div id="stage"><canvas id="cv" width="520" height="520"></canvas></div>
  <aside>
    <div id="classes"></div>
    <div class="row">
      <button id="prev">← 上一张</button>
      <button id="next">下一张 →</button>
    </div>
    <div class="row">
      <button id="undo">撤销 (Backspace)</button>
    </div>
    <div class="row">
      <button class="primary" id="save">导出 JSON</button>
    </div>
    <div class="hint">
      <b>1–6</b> 选类别　<b>拖拽</b> 画框<br>
      <b>Backspace</b> 删除最后画的一个框<br>
      <b>← →</b> 切换图片　<b>回车</b> 下一张<br>
      画完点“导出 JSON”，把下载的文件发回来。
    </div>
  </aside>
</main>
<script>
const IMAGES = __IMAGES__;
const CLASSES = __CLASSES__;
const COLORS = __COLORS__;
const KEY = "neudet-manual-annotations-v1";

let store = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = 0, cls = 0;
let img = new Image(), drag = null;

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");

function boxes(name) { return store[name] || (store[name] = []); }
function persist() { localStorage.setItem(KEY, JSON.stringify(store)); }

function renderClasses() {
  const host = document.getElementById("classes");
  host.innerHTML = "";
  CLASSES.forEach((c, i) => {
    const d = document.createElement("div");
    d.className = "cls" + (i === cls ? " on" : "");
    d.innerHTML = `<span class="key" style="background:${COLORS[i]}">${i + 1}</span>
                   <span>${c}</span><span class="n">${countClass(i)}</span>`;
    d.onclick = () => { cls = i; renderClasses(); };
    host.appendChild(d);
  });
}
function countClass(i) {
  return Object.values(store).flat().filter(b => b[0] === i).length;
}

function draw() {
  const S = cv.width / img.width;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const list = boxes(IMAGES[idx].name);
  list.forEach(([c, x, y, w, h]) => {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = COLORS[c];
    ctx.fillStyle = COLORS[c] + "22";
    ctx.fillRect(x * S, y * S, w * S, h * S);
    ctx.strokeRect(x * S, y * S, w * S, h * S);
    const tag = CLASSES[c];
    ctx.font = "13px Arial";
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = COLORS[c];
    ctx.fillRect(x * S, Math.max(0, y * S - 18), tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, x * S + 5, Math.max(12, y * S - 5));
  });
  if (drag) {
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = COLORS[cls];
    const x = Math.min(drag.x0, drag.x1), y = Math.min(drag.y0, drag.y1);
    const w = Math.abs(drag.x1 - drag.x0), h = Math.abs(drag.y1 - drag.y0);
    ctx.strokeRect(x * S, y * S, w * S, h * S);
    ctx.setLineDash([]);
  }
  document.getElementById("prog").textContent = `${idx + 1} / ${IMAGES.length}`;
  document.getElementById("prog").className =
    "progress" + (boxes(IMAGES[idx].name).length ? " done" : "");
  renderClasses();
}

function load(i) {
  idx = Math.max(0, Math.min(IMAGES.length - 1, i));
  img = new Image();
  img.onload = draw;
  img.src = IMAGES[idx].src;
}

function pos(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left) * img.width / r.width,
           y: (e.clientY - r.top) * img.height / r.height };
}

cv.addEventListener("mousedown", e => { const p = pos(e); drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; });
cv.addEventListener("mousemove", e => { if (!drag) return; const p = pos(e); drag.x1 = p.x; drag.y1 = p.y; draw(); });
window.addEventListener("mouseup", e => {
  if (!drag) return;
  const x = Math.min(drag.x0, drag.x1) / img.width, y = Math.min(drag.y0, drag.y1) / img.height;
  const w = Math.abs(drag.x1 - drag.x0) / img.width, h = Math.abs(drag.y1 - drag.y0) / img.height;
  if (w > 0.01 && h > 0.01) { boxes(IMAGES[idx].name).push([cls, x, y, w, h]); persist(); }
  drag = null; draw();
});

document.getElementById("prev").onclick = () => load(idx - 1);
document.getElementById("next").onclick = () => load(idx + 1);
document.getElementById("undo").onclick = () => { boxes(IMAGES[idx].name).pop(); persist(); draw(); };
document.getElementById("save").onclick = () => {
  const payload = { classes: CLASSES, images: {} };
  IMAGES.forEach(im => { payload.images[im.name] = store[im.name] || []; });
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations.json";
  a.click();
};

window.addEventListener("keydown", e => {
  if (e.key >= "1" && e.key <= String(CLASSES.length)) { cls = +e.key - 1; renderClasses(); }
  else if (e.key === "Backspace") { e.preventDefault(); boxes(IMAGES[idx].name).pop(); persist(); draw(); }
  else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); load(idx + 1); }
  else if (e.key === "ArrowLeft") { load(idx - 1); }
});

load(0);
</script>
</body>
</html>
"""


def main():
    imgs = sorted((TASK / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit("no images in annotation_task/images")

    payload = []
    for p in imgs:
        b64 = base64.b64encode(p.read_bytes()).decode()
        payload.append({"name": p.name, "src": f"data:image/jpeg;base64,{b64}"})

    html = (
        PAGE.replace("__IMAGES__", json.dumps(payload))
        .replace("__CLASSES__", json.dumps(CLASSES, ensure_ascii=False))
        .replace("__COLORS__", json.dumps(COLORS))
    )
    out = TASK / "annotate.html"
    out.write_text(html, encoding="utf-8")
    print(f"built {out} ({out.stat().st_size / 1e6:.1f} MB, {len(payload)} images inlined)")


if __name__ == "__main__":
    main()
