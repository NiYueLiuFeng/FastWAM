"""viz_v3_html.py — 用浏览器可视化 LeRobot v3.0 数据集（网页版，零外部依赖）。

参考 lerobot v2.1 的 visualize_dataset_html.py（Flask + dygraph + CDN），针对 v3.0
格式与当前环境重写，主要差异：

  - v3.0 的视频是多 episode 拼接的大 mp4（meta/episodes parquet 记录每个 episode
    的 from/to_timestamp），本脚本用 HTML5 media fragment `#t=from,to` +
    loadedmetadata 兜底 seek，把浏览器播放器锚定到该 episode 的视频段；
  - venv 无 flask → 用标准库 http.server（带 Range 请求支持，这是浏览器对
    拼接 mp4 seek 的必要条件）；
  - 不依赖外网 CDN → 曲线用零依赖 vanilla JS canvas 绘制（按字段分面板）；
  - 曲线时间轴取 frame_index / fps（与 LeRobotDataset 按 from_timestamp +
    k/fps 解码视频帧的锚定方式一致），图像走浏览器原生 mp4 解码 —— 两条数据源
    在同一页面按时间对齐，页面顶部的"对齐检查"提示即是对齐性核验。

页面布局（仿参考脚本）：左侧 episode 列表；右侧上方各路相机视频（同步播放/拖动/
倍速），下方每个数值字段一个曲线面板 —— 1-D 字段直接画各维；2-D 字段（如手部
21 关节 × xyz/四元数）默认画第一维索引 0 的分量，面板标题栏的选择框可切换 0~n，
选择结果与字段勾选一样存 localStorage。红色竖线 = 当前视频时间；面板悬停查看
该帧各维数值，点击 seek 到对应时间。
快捷键：Space 播放/暂停，←/→ 单帧步进，↑/↓ 切换 episode。

用法:
    python scripts/viz_v3_html.py --root <数据集根目录>                     # 本机查看
    python scripts/viz_v3_html.py --root ... --host 0.0.0.0 --port 9070    # 远程机器
    # 远程时本机浏览器打开 http://<远程IP>:9070，或 ssh -L 9070:localhost:9070 <远程>

依赖: pyarrow, numpy —— venv 均已具备。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>__REPO__ · episode __EP__</title>
<style>
  body { margin:0; display:flex; height:100vh; background:#0f172a; color:#e2e8f0;
         font-family: ui-monospace, Consolas, monospace; overflow:hidden; }
  #side { width:230px; min-width:230px; background:#1e293b; overflow-y:auto; padding:12px;
          border-right:1px solid #334155; }
  #side h2 { font-size:14px; margin:4px 0 8px; word-break:break-all; }
  #side .meta { font-size:11px; color:#94a3b8; margin-bottom:10px; line-height:1.6;
                word-break:break-all; }
  #side a { color:#7dd3fc; text-decoration:none; font-size:12px; display:block;
            padding:1px 4px; border-radius:3px; }
  #side a.cur { background:#0ea5e9; color:#0f172a; font-weight:bold; }
  #side a:hover:not(.cur) { background:#334155; }
  #main { flex:1; overflow-y:auto; padding:14px 18px; }
  .banner { background:#7c2d12; color:#fed7aa; padding:8px 12px; border-radius:6px;
            font-size:13px; margin-bottom:8px; display:none; }
  .note  { background:#1e3a5f; color:#bfdbfe; padding:6px 12px; border-radius:6px;
            font-size:12px; margin-bottom:8px; display:none; }
  #vidrow { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 6px; }
  .vbox { position:relative; margin-top:16px; }
  .vbox p { position:absolute; top:-15px; left:2px; font-size:11px; color:#94a3b8; }
  video { width:320px; height:240px; background:#000; border-radius:4px; object-fit:contain; }
  #ctrl { display:flex; gap:6px; align-items:center; margin:10px 0 14px; font-size:13px;
          flex-wrap:wrap; }
  #ctrl button { background:#334155; color:#e2e8f0; border:none; border-radius:5px;
                 padding:5px 10px; cursor:pointer; font-size:14px; }
  #ctrl button:hover { background:#475569; }
  #ctrl input[type=range] { width:min(420px, 50vw); }
  #timer { border:1px solid #475569; border-radius:5px; padding:2px 8px; font-size:12px; }
  .panel { position:relative; margin-bottom:16px; }
  .panel .phead { display:flex; align-items:center; gap:8px; margin:0 0 2px 2px; }
  .panel .phead h3 { font-size:12px; color:#cbd5e1; margin:0; font-weight:normal; }
  .panel .phead select { background:#1e293b; color:#e2e8f0; border:1px solid #334155;
                         border-radius:4px; padding:1px 4px; font-size:11px; }
  .panel canvas { width:100%; height:130px; display:block; background:#101418;
                  border:1px solid #334155; border-radius:4px; cursor:crosshair; }
  .tip { position:absolute; pointer-events:none; background:#000c; color:#e2e8f0;
         border:1px solid #475569; border-radius:4px; padding:6px 8px; font-size:11px;
         line-height:1.5; white-space:pre; z-index:5; display:none; }
  #fieldbar { display:flex; flex-wrap:wrap; align-items:center; gap:4px; margin:0 0 12px; }
  #fieldbar .ftitle { font-size:12px; color:#94a3b8; margin-right:4px; }
  #fieldbar button { background:#334155; color:#e2e8f0; border:none; border-radius:4px;
                     padding:3px 8px; cursor:pointer; font-size:11px; }
  #fieldbar button:hover { background:#475569; }
  #fieldbar label { display:inline-flex; align-items:center; gap:4px; background:#1e293b;
                    border:1px solid #334155; padding:3px 9px; border-radius:12px;
                    font-size:11px; cursor:pointer; user-select:none; }
  #fieldbar label:hover { border-color:#0ea5e9; }
  #fieldbar input { accent-color:#0ea5e9; margin:0; }
</style>
</head>
<body>
<div id="side">
  <h2>__REPO__</h2>
  <div class="meta" id="meta"></div>
  <div id="eplist"></div>
</div>
<div id="main">
  <div class="banner" id="codecBanner">
    浏览器可能不支持 AV1 解码，视频无法播放（曲线不受影响）。
    建议使用较新版本的 Chrome / Edge / Firefox。
  </div>
  <div class="note" id="alignNote"></div>
  <div id="vidrow"></div>
  <div id="ctrl">
    <button id="btnPlay" title="Space 播放/暂停">▶</button>
    <button id="btnBack" title="后退 1 帧 (←)">◀|</button>
    <button id="btnFwd" title="前进 1 帧 (→)">|▶</button>
    <input type="range" id="slider" min="0" max="1000" value="0">
    <span id="timer">0.00 / 0.00 s</span>
    <select id="speed">
      <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
      <option value="1" selected>1x</option><option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <span style="color:#64748b;font-size:11px">Space 播放/暂停 · ←/→ 步进 · ↑/↓ 切集</span>
  </div>
  <div id="fieldbar">
    <span class="ftitle">曲线字段:</span>
    <button id="fAll" title="显示全部字段">全选</button>
    <button id="fNone" title="只看视频">清空</button>
  </div>
  <div id="panels"></div>
</div>
<script>
const EP = __PAYLOAD__;

// ---------- 解析内嵌 CSV（timestamp + 各字段摊平列，按 fields[].cols 顺序） ----------
const lines = EP.csv.split("\n");
const rows = [];
for (let i = 1; i < lines.length; i++) {
  if (!lines[i]) continue;
  const v = lines[i].split(",");
  const r = new Float64Array(v.length);
  for (let j = 0; j < v.length; j++) r[j] = parseFloat(v[j]);
  rows.push(r);
}
const times = rows.map(r => r[0]);
const DUR = times.length ? times[times.length - 1] : 0;
const dur = Math.max(DUR, 1e-6);

// ---------- 侧栏 ----------
document.getElementById("meta").innerHTML =
  `frames: ${EP.length}<br>fps: ${EP.fps}<br>episodes: ${EP.episodes.length}` +
  (EP.task ? `<br>task: ${EP.task}` : "");
const epList = document.getElementById("eplist");
for (const e of EP.episodes) {
  const a = document.createElement("a");
  a.href = "/episode/" + e;
  a.textContent = "episode " + e;
  if (e === EP.episode_id) a.className = "cur";
  epList.appendChild(a);
}
if (EP.notes.length) {
  const n = document.getElementById("alignNote");
  n.style.display = "block";
  n.innerHTML = "对齐检查: " + EP.notes.join("；");
}

// ---------- 视频（media fragment 锚定到该 episode 的段） ----------
const vidrow = document.getElementById("vidrow");
const videos = [];
for (const v of EP.videos) {
  const box = document.createElement("div");
  box.className = "vbox";
  const label = document.createElement("p");
  label.textContent = v.key;
  const el = document.createElement("video");
  el.muted = true; el.preload = "auto"; el.loop = true;
  el.src = v.url + "#t=" + v.from + "," + v.to;
  box.appendChild(label); box.appendChild(el); vidrow.appendChild(box);
  el.addEventListener("loadedmetadata", () => {
    // media fragment 不生效时的兜底：手动 seek 到段起点
    if (Math.abs(el.currentTime - v.from) > 0.05) el.currentTime = v.from;
  });
  el.addEventListener("error", () => {
    document.getElementById("codecBanner").style.display = "block";
  });
  videos.push({ el, from: v.from, key: v.key });
}
const master = videos.length ? videos[0] : null;

function allSeek(t) {  // t = episode 内数据时间（0 起）
  t = Math.max(0, Math.min(t, dur));
  for (const v of videos) v.el.currentTime = v.from + t;
  updateUI(t);
}
function curT() {
  return master ? Math.max(0, Math.min(master.el.currentTime - master.from, dur)) : 0;
}
if (master) {
  master.el.addEventListener("timeupdate", () => {
    // 漂移校正必须按"数据时间"比较：各相机在不同拼接文件、不同绝对时间轴上，
    // 直接比较 currentTime 会把别的相机拽到自己文件里同一绝对时刻（= 错误的
    // episode 内容）。正确做法是各自减去自己的 from 再比较。
    const mt = master.el.currentTime - master.from;
    for (const v of videos) {
      if (v === master) continue;
      const vt = v.el.currentTime - v.from;
      if (Math.abs(vt - mt) > 0.1)  // ~3 帧内不纠正，避免播放中频繁 seek
        v.el.currentTime = v.from + Math.max(0, Math.min(mt, dur));
    }
    updateUI(curT());
  });
  master.el.addEventListener("seeked", () => {
    const mt = master.el.currentTime - master.from;
    for (const v of videos)
      if (v !== master)
        v.el.currentTime = v.from + Math.max(0, Math.min(mt, dur));
    updateUI(curT());
  });
}

// ---------- 控件 ----------
const slider = document.getElementById("slider");
const timer = document.getElementById("timer");
const btnPlay = document.getElementById("btnPlay");
btnPlay.onclick = () => {
  const p = master ? master.el.paused : true;
  for (const v of videos) p ? v.el.play() : v.el.pause();
  btnPlay.textContent = p ? "⏸" : "▶";
};
document.getElementById("btnFwd").onclick = () => allSeek(curT() + 1 / EP.fps);
document.getElementById("btnBack").onclick = () => allSeek(curT() - 1 / EP.fps);
slider.oninput = () => allSeek(dur * slider.value / 1000);
document.getElementById("speed").onchange = (e) =>
  videos.forEach(v => v.el.playbackRate = parseFloat(e.target.value));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); btnPlay.click(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); allSeek(curT() + 1 / EP.fps); }
  else if (e.key === "ArrowLeft")  { e.preventDefault(); allSeek(curT() - 1 / EP.fps); }
  else if (e.key === "ArrowDown")  { e.preventDefault(); nav(1); }
  else if (e.key === "ArrowUp")    { e.preventDefault(); nav(-1); }
});
function nav(d) {
  const i = EP.episodes.indexOf(EP.episode_id);
  if (i >= 0 && i + d >= 0 && i + d < EP.episodes.length)
    location.href = "/episode/" + EP.episodes[i + d];
}

// ---------- 曲线面板（每个字段一块 canvas；静态曲线离屏缓存，逐帧只叠游标） ----------
const palette = [];
for (let i = 0; i < 32; i++)
  palette.push(`hsl(${(i * 137) % 360}, 85%, ${58 + (i % 3) * 10}%)`);

const panels = [];
const panelDiv = document.getElementById("panels");

// 二维字段（d0>1）的第一维索引选择也存 localStorage（字段 key → 索引）
const LS_JOINTS = "viz_v3_html_joints";
let savedJoints = null;
try {
  if (typeof localStorage !== "undefined")
    savedJoints = JSON.parse(localStorage.getItem(LS_JOINTS) || "null");
} catch (e) { /* localStorage 不可用时忽略 */ }
function saveJoints() {
  try {
    if (typeof localStorage !== "undefined") {
      const m = {};
      for (const p of panels) if (p.f.d0 > 1) m[p.f.key] = p.joint;
      localStorage.setItem(LS_JOINTS, JSON.stringify(m));
    }
  } catch (e) { /* 忽略 */ }
}

for (let fi = 0; fi < EP.fields.length; fi++) {
  const f = EP.fields[fi];
  const box = document.createElement("div");
  box.className = "panel";
  const head = document.createElement("div");
  head.className = "phead";
  const h = document.createElement("h3");
  h.textContent = f.key + "  (" +
    (f.d0 > 1 ? f.d0 + "×" + f.dims.length : f.dims.length + " dims") + ")";
  head.appendChild(h);
  const cvs = document.createElement("canvas");
  const tip = document.createElement("div");
  tip.className = "tip";
  box.appendChild(head); box.appendChild(cvs); box.appendChild(tip);
  panelDiv.appendChild(box);
  // 该字段的数据列在 CSV 中的起始位置（列 0 是 timestamp；二维字段是整块摊平）
  let off = 1;
  for (let k = 0; k < fi; k++) off += EP.fields[k].cols.length;
  const p = { f, off, box, cvs, tip, joint: 0, data: [], lo: 0, hi: 1 };
  // 二维字段：标题栏加第一维索引选择框，默认 0；head_world 等 d0==1 的没有
  if (f.d0 > 1) {
    const sel = document.createElement("select");
    for (let j = 0; j < f.d0; j++) {
      const o = document.createElement("option");
      o.value = j;
      o.textContent = f.joints ? f.joints[j] : "#" + j;
      sel.appendChild(o);
    }
    if (savedJoints && savedJoints[f.key] !== undefined)
      p.joint = Math.max(0, Math.min(savedJoints[f.key] | 0, f.d0 - 1));
    sel.value = p.joint;
    sel.onchange = () => {
      p.joint = parseInt(sel.value, 10);
      saveJoints();
      slicePanel(p);
      if (p.box.style.display !== "none") { drawPanelBase(p); updateUI(curT()); }
    };
    head.appendChild(sel);
  }
  panels.push(p);
}

// 面板数据 = 当前第一维索引下的 dims.length 条列，范围随之重算（不重画）
function slicePanel(p) {
  const d1 = p.f.dims.length, off = p.off + p.joint * d1;
  const data = [];
  for (let d = 0; d < d1; d++) data.push(rows.map(r => r[off + d]));
  let lo = Infinity, hi = -Infinity;
  for (const col of data) for (const x of col) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05;
  p.data = data; p.lo = lo - pad; p.hi = hi + pad;
}
for (const p of panels) slicePanel(p);

// ---------- 字段勾选：默认只显示视频，勾选后出现对应曲线面板 ----------
// 选择存 localStorage，切换 episode / 刷新后保持。
const LS_KEY = "viz_v3_html_fields";
let savedFields = null;
try {
  if (typeof localStorage !== "undefined")
    savedFields = JSON.parse(localStorage.getItem(LS_KEY) || "null");
} catch (e) { /* localStorage 不可用时忽略 */ }
const fieldbar = document.getElementById("fieldbar");
function saveFields() {
  try {
    if (typeof localStorage !== "undefined")
      localStorage.setItem(LS_KEY, JSON.stringify(
        panels.filter(p => p.box.style.display !== "none").map(p => p.f.key)));
  } catch (e) { /* 忽略 */ }
}
function setPanel(p, on) {
  p.box.style.display = on ? "" : "none";
  if (on) { drawBase(); updateUI(curT()); }  // 显示时按实际宽度重画
}
for (const p of panels) {
  const lab = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = savedFields ? savedFields.includes(p.f.key) : false;  // 默认全不选
  cb.onchange = () => { setPanel(p, cb.checked); saveFields(); };
  const span = document.createElement("span");
  span.textContent = p.f.key;
  lab.appendChild(cb); lab.appendChild(span); fieldbar.appendChild(lab);
  p.cb = cb;
  p.box.style.display = cb.checked ? "" : "none";
}
document.getElementById("fAll").onclick = () => {
  panels.forEach(p => { p.cb.checked = true; setPanel(p, true); });
  saveFields();
};
document.getElementById("fNone").onclick = () => {
  panels.forEach(p => { p.cb.checked = false; setPanel(p, false); });
  saveFields();
};

function drawPanelBase(p) {  // 单面板全量曲线画进离屏画布
  const dpr = window.devicePixelRatio || 1;
  const w = p.cvs.clientWidth || 900, hgt = p.cvs.clientHeight || 130;
  if (!p.base) p.base = document.createElement("canvas");
  p.base.width = w * dpr; p.base.height = hgt * dpr;
  p.cvs.width = w * dpr; p.cvs.height = hgt * dpr;
  p._w = w; p._h = hgt; p._dpr = dpr;
  const c = p.base.getContext("2d");
  c.scale(dpr, dpr);
  c.fillStyle = "#101418"; c.fillRect(0, 0, w, hgt);
  c.strokeStyle = "#28323a"; c.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    const y = (hgt * g) / 4;
    c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke();
  }
  c.fillStyle = "#64748b"; c.font = "10px monospace";
  c.fillText(p.lo.toFixed(2), 2, 10);
  c.fillText(p.hi.toFixed(2), 2, hgt - 3);
  const x = (i) => (times[i] / dur) * w;
  const y = (v) => hgt - 4 - ((v - p.lo) / (p.hi - p.lo)) * (hgt - 8);
  for (let d = 0; d < p.data.length; d++) {
    c.strokeStyle = palette[d % palette.length]; c.lineWidth = 1;
    c.beginPath();
    for (let i = 0; i < times.length; i++) {
      const px = x(i), py = y(p.data[d][i]);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    }
    c.stroke();
  }
}

function drawBase() {  // 所有可见面板重画（resize / 字段显隐 / 切关节时）
  for (const p of panels)
    if (p.box.style.display !== "none") drawPanelBase(p);  // 隐藏面板不画
}

function updateUI(t) {
  slider.value = Math.round((t / dur) * 1000);
  timer.textContent = t.toFixed(2) + " / " + dur.toFixed(2) + " s";
  for (const p of panels) {
    if (p.box.style.display === "none" || !p.base) continue;  // 隐藏/未绘制
    const c = p.cvs.getContext("2d");
    c.clearRect(0, 0, p.cvs.width, p.cvs.height);
    c.drawImage(p.base, 0, 0);
    const px = (t / dur) * p._w * p._dpr;
    c.strokeStyle = "#ff2d2d"; c.lineWidth = 2 * p._dpr;
    c.beginPath(); c.moveTo(px, 0); c.lineTo(px, p._h * p._dpr); c.stroke();
  }
}

// 面板交互：悬停显示该帧各维数值，点击 seek
for (const p of panels) {
  p.cvs.addEventListener("mousemove", (e) => {
    const r = p.cvs.getBoundingClientRect();
    const t = Math.max(0, Math.min((e.clientX - r.left) / r.width * dur, dur));
    let i = 0;
    while (i + 1 < times.length && times[i + 1] < t) i++;
    const jname = p.f.d0 > 1 ? (p.f.joints ? p.f.joints[p.joint] : "#" + p.joint) : null;
    let txt = `t=${times[i].toFixed(3)}s  frame ${i}` + (jname ? `  [${jname}]` : "");
    for (let d = 0; d < p.f.dims.length; d++)
      txt += `\n${d}:${p.f.dims[d]} = ${p.data[d][i].toFixed(4)}`;
    p.tip.style.display = "block";
    p.tip.textContent = txt;
    p.tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 190) + "px";
    p.tip.style.top = (e.clientY - r.top - 20) + "px";
  });
  p.cvs.addEventListener("mouseleave", () => p.tip.style.display = "none");
  p.cvs.addEventListener("click", (e) => {
    const r = p.cvs.getBoundingClientRect();
    allSeek(Math.max(0, Math.min((e.clientX - r.left) / r.width * dur, dur)));
  });
}

window.addEventListener("resize", () => { drawBase(); updateUI(curT()); });
drawBase();
updateUI(0);

// URL ?t=<秒>：等视频元数据就绪后 seek 到起始时刻
const q = new URLSearchParams(location.search).get("t");
if (q && master) master.el.addEventListener("loadedmetadata",
  () => allSeek(parseFloat(q)), { once: true });

// AV1 能力检查（参考 lerobot v2.1 模板的做法）
if (!document.createElement("video").canPlayType('video/mp4; codecs="av01.0.08M.08"'))
  document.getElementById("codecBanner").style.display = "block";
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 数据侧：读 v3.0 的 meta/info.json + meta/episodes parquet + data parquet
# ---------------------------------------------------------------------------

def load_episodes(root: Path) -> dict[int, dict]:
    """meta/episodes parquet → {ep: {length, tasks, videos: {key: {...}}}}。"""
    eps: dict[int, dict] = {}
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise SystemExit(f"未找到 {root}/meta/episodes/chunk-*/file-*.parquet（不是 v3.0 布局？）")
    for f in files:
        tbl = pq.read_table(f)
        cols = tbl.column_names
        vkeys = sorted({c.split("/")[1] for c in cols if c.startswith("videos/")})
        for row in tbl.to_pylist():
            ep = row["episode_index"]
            videos = {}
            for k in vkeys:
                p = f"videos/{k}/"
                videos[k] = {
                    "chunk": row[p + "chunk_index"],
                    "file": row[p + "file_index"],
                    "from": row[p + "from_timestamp"],
                    "to": row[p + "to_timestamp"],
                }
            tasks = row.get("tasks") or []
            eps[ep] = {"length": row["length"], "tasks": tasks, "videos": videos}
    return eps


def _dim_names(feat: dict, key: str) -> list[str]:
    """维度名：优先 meta features 的 names，否则用 字段末段_i。"""
    names = feat.get("names")
    if isinstance(names, list) and len(names) == feat["shape"][0]:
        return [str(n) for n in names]
    return [f"{key.split('.')[-1]}_{i}" for i in range(feat["shape"][0])]


_COMPONENTS = {3: ("x", "y", "z"), 4: ("qw", "qx", "qy", "qz")}


def _field_desc(feat: dict, key: str) -> dict:
    """字段的曲线描述（进 EP.fields，驱动前端面板）。

    1-D 字段：面板画 shape[0] 条曲线，列名沿用 _dim_names。
    2-D 字段 (d0, d1)：面板画选中的第一维索引的 d1 条分量曲线（默认 0，前端
    选择框可切换 0~d0-1）；CSV 列为整字段摊平（d0*d1 列），列名
    ``{末段}_{j}_{分量}``，分量名按 d1 取 x/y/z 或四元数 qw/qx/qy/qz。
    """
    shape = feat.get("shape", [])
    if len(shape) == 2:
        d0, d1 = shape
        comps = list(_COMPONENTS.get(d1, [f"c{i}" for i in range(d1)]))
        last = key.split(".")[-1]
        names = feat.get("names")
        joints = ([str(n) for n in names]
                  if isinstance(names, list) and len(names) == d0
                  else [str(j) for j in range(d0)])
        return {
            "key": key,
            "dims": comps,
            "d0": d0,
            "joints": joints,
            "cols": [f"{last}_{j}_{c}" for j in range(d0) for c in comps],
        }
    dims = _dim_names(feat, key)
    return {"key": key, "dims": dims, "d0": 1, "cols": dims}


@lru_cache(maxsize=8)
def episode_payload(root_str: str, ep: int, fps: int) -> dict:
    """读一个 episode 的 state/action parquet 行 + 视频段信息，组装页面数据。"""
    root = Path(root_str)
    feats = json.load(open(root / "meta/info.json"))["features"]
    ep_meta = _EPISODES[root_str][ep]

    # 数值型字段进曲线：1-D 画整字段，2-D（如 21 关节 × xyz）由前端选第一维索引；
    # video 与 3D+ 不进
    fields = [k for k, v in feats.items()
              if v.get("dtype") in ("float32", "float64", "int32", "int64")
              and len(v.get("shape", [])) in (1, 2)
              and k not in ("timestamp", "frame_index", "episode_index",
                            "index", "task_index")]
    fdescs = [_field_desc(feats[k], k) for k in fields]

    # 从 data parquet 读该 episode 的行（filters 只扫相关 row group）
    tbls = []
    for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=fields + ["frame_index", "episode_index"],
                          filters=[("episode_index", "=", ep)])
        if t.num_rows:
            tbls.append(t)
    if not tbls:
        raise KeyError(f"data parquet 中没有 episode {ep} 的行")
    tbl = pa.concat_tables(tbls).sort_by("frame_index")
    n = tbl.num_rows

    # 时间轴 = frame_index / fps（与视频解码锚定 from + k/fps 一致）
    ts = [i / fps for i in tbl.column("frame_index").to_pylist()]
    # 各字段摊平为 (n, cols)：1-D → (n, d)，2-D → (n, d0*d1)
    mats = [np.asarray(tbl.column(k).to_pylist(), dtype=np.float64).reshape(n, -1)
            for k in fields]

    header = ["timestamp"]
    for d in fdescs:
        header += d["cols"]
    lines = [",".join(header)]
    for i in range(n):
        parts = [f"{ts[i]:.4f}"] + [f"{x:.6g}" for m in mats for x in m[i]]
        lines.append(",".join(parts))

    # 视频段（URL 由本服务的 /videos 路径提供，带 Range 支持）
    videos, notes = [], []
    for k, v in sorted(ep_meta["videos"].items()):
        p = root / "videos" / k / f"chunk-{v['chunk']:03d}" / f"file-{v['file']:03d}.mp4"
        if not p.exists():
            notes.append(f"{k.split('.')[-1]}: 视频文件缺失")
            continue
        videos.append({
            "key": k,
            "url": f"/videos/{quote(k)}/chunk-{v['chunk']:03d}/file-{v['file']:03d}.mp4",
            "from": v["from"], "to": v["to"],
        })
        # 对齐检查：视频段长度 vs 数据帧数换算长度
        seg = v["to"] - v["from"]
        want = n / fps
        if abs(seg - want) > 0.2:
            notes.append(f"{k.split('.')[-1]}: 视频段 {seg:.2f}s vs 数据 {want:.2f}s")
    if n != ep_meta["length"]:
        notes.append(f"parquet 行数 {n} != meta 登记长度 {ep_meta['length']}")
    if not videos:
        notes.append("没有可用视频（仅曲线）")

    return {
        "csv": "\n".join(lines),
        "fields": fdescs,
        "videos": videos,
        "length": n,
        "notes": notes,
        "task": (ep_meta["tasks"][0] if ep_meta["tasks"] else None),
    }


# ---------------------------------------------------------------------------
# HTTP 服务（标准库，支持 Range —— 浏览器对拼接 mp4 seek 的必要条件）
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root: Path = None
    nav: list = None
    fps: int = 30
    repo_id: str = ""

    def log_message(self, fmt, *args):  # 视频分段请求频繁，只记非视频日志
        if "/videos/" not in (fmt % args if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path in ("/", ""):
                return self._redirect(f"/episode/{self.nav[0]}")
            m = re.fullmatch(r"/episode/(\d+)", path)
            if m:
                return self._page(int(m.group(1)))
            if path.startswith("/videos/"):
                return self._video(path[len("/videos/"):])
            self.send_error(404, "not found")
        except KeyError:
            self.send_error(404, "episode not found")
        except BrokenPipeError:
            pass

    def _redirect(self, to):
        # HTTP/1.1 keep-alive 下必须声明长度/关闭，否则客户端会一直等
        self.send_response(302)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.close_connection = True
        self.end_headers()

    def _page(self, ep: int):
        payload = episode_payload(str(self.root), ep, self.fps)
        payload.update({"episode_id": ep, "episodes": self.nav,
                        "fps": self.fps, "repo_id": self.repo_id})
        body = (PAGE
                .replace("__REPO__", self.repo_id)
                .replace("__EP__", str(ep))
                .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")))
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video(self, rel: str):
        p = (self.root / "videos" / rel).resolve()
        if not p.is_relative_to((self.root / "videos").resolve()) or not p.is_file():
            return self.send_error(404, "video not found")
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


# 每个 root 的 episode 元数据（启动时加载一次，供 lru_cache 的 payload 使用）
_EPISODES: dict[str, dict[int, dict]] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="数据集根目录（含 meta/info.json）")
    ap.add_argument("--repo-id", default=None, help="显示用标签；默认取 root 目录名")
    ap.add_argument("--episodes", default=None, help="逗号分隔的 episode id；默认全部")
    ap.add_argument("--host", default="127.0.0.1", help="远程机器用 0.0.0.0")
    ap.add_argument("--port", type=int, default=9070)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    info = json.load(open(args.root / "meta/info.json"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        logging.warning("codebase_version=%s 不是 v3.x，布局可能有出入，继续尝试…",
                        info.get("codebase_version"))
    eps = load_episodes(args.root)
    _EPISODES[str(args.root)] = eps

    nav = sorted(eps) if not args.episodes else \
        [int(x) for x in args.episodes.split(",") if x.strip()]
    invalid = [e for e in nav if e not in eps]
    if invalid:
        logging.warning("episode %s 不存在，已忽略", invalid)
        nav = [e for e in nav if e in eps]
    if not nav:
        raise SystemExit("没有可用的 episode")

    Handler.root = args.root
    Handler.nav = nav
    Handler.fps = info["fps"]
    Handler.repo_id = args.repo_id or args.root.name

    host = "localhost" if args.host == "127.0.0.1" else args.host
    logging.info("%s: %s episodes / %s frames / fps=%s；episode %d … %d",
                 args.root.name, info["total_episodes"], info["total_frames"],
                 info["fps"], nav[0], nav[-1])
    logging.info("浏览器打开 http://%s:%d（远程机器加 --host 0.0.0.0，"
                 "或 ssh -L %d:localhost:%d）", host, args.port, args.port, args.port)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nCtrl-C 收到，退出。")


if __name__ == "__main__":
    main()
