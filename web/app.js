/* Ka-band Video SAR shadow workbench — single-page flow: 输入 → 模型 → 结果.
 * No framework, no CDN; works fully offline. */
"use strict";

const $ = (id) => document.getElementById(id);

const METHODS = {
  threshold: { label: "局部暗度阈值", hint: "传统 · 无需训练" },
  difference: { label: "时序背景差分", hint: "传统 · 抵消静态背景" },
  learned: { label: "U-Net 时序", hint: "学习模型 · 需 PyTorch" },
};

const METRIC_ROWS = [
  { key: "target_f1", label: "目标级 F1", better: "up", digits: 3 },
  { key: "iou", label: "区域 IoU", better: "up", digits: 3 },
  { key: "precision", label: "查准率", better: "up", digits: 3 },
  { key: "recall", label: "召回率", better: "up", digits: 3 },
  { key: "fp_per_frame", label: "每帧虚警", better: "down", digits: 2 },
  { key: "matched", label: "匹配帧", better: "up", digits: 0 },
  { key: "missed", label: "漏检帧", better: "down", digits: 0 },
  { key: "continuity", label: "轨迹连续性", better: "up", digits: 3 },
  { key: "location_rmse_px", label: "定位 RMSE (px)", better: "down", digits: 2 },
  { key: "velocity_error_mps", label: "速度误差 (m/s)", better: "down", digits: 3 },
];

const state = {
  boot: null,
  runsRoot: "",
  kind: "sequence",
  seq: "",
  video: "",
  videoInfo: null,
  cropImg: null,
  crop: { x: 0, y: 0, w: 0, h: 0 },
  dragging: false,
  dragStart: null,
  methods: new Set(["threshold", "learned"]),
  results: [],
  meta: null,
  count: 0,
  index: 0,
  view: "compare",
  playing: false,
  timer: null,
  quadCache: new Map(),
  active: 0,
  videoResult: null,
};

/* ---------------------------------------------------------------- helpers */
async function apiJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

function escapeHtml(val) {
  return String(val).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function setOptions(selectEl, items, valueKey = "path", labelKey = "label") {
  selectEl.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    const isObj = typeof item === "object";
    opt.value = isObj ? item[valueKey] : item;
    opt.textContent = isObj ? item[labelKey] : item;
    selectEl.appendChild(opt);
  }
}

function renderTable(container, fields, rows) {
  if (!fields || !fields.length || !rows || !rows.length) {
    container.innerHTML = `<div style="padding:16px;color:var(--text-muted);font-size:12.5px;">暂无数据</div>`;
    return;
  }
  const head = `<thead><tr>${fields.map((f) => `<th>${escapeHtml(f)}</th>`).join("")}</tr></thead>`;
  const body = rows
    .map((r) => `<tr>${fields.map((f) => `<td class="mono">${escapeHtml(r[f] ?? "")}</td>`).join("")}</tr>`)
    .join("");
  container.innerHTML = `<table class="plain">${head}<tbody>${body}</tbody></table>`;
}

function getSettings() {
  return {
    mode: $("dispMode").value || "percentile",
    cmap: $("dispCmap").value || "gray",
    denoise: $("dispDenoise").checked,
    diff_ref: $("dispDiff").value || "median",
    show_truth: $("dispTruth").checked,
    show_pred: $("dispPred").checked,
  };
}

function settingsKey() {
  const s = getSettings();
  return `${s.mode}|${s.cmap}|${s.denoise}|${s.diff_ref}|${s.show_truth}|${s.show_pred}`;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return digits === 0 ? String(Math.round(value)) : value.toFixed(digits);
}

function labelOf(result) {
  if (result.method && METHODS[result.method]) return METHODS[result.method].label;
  return result.label || "已有关联结果";
}

function setBadge(el, text, level = "") {
  el.textContent = text;
  el.className = `badge ${level}`.trim();
}

/* ---------------------------------------------------------------- context */
function updateCtx() {
  const chip = $("ctxChip");
  const parts = [];
  if (state.kind === "video") {
    if (state.video) parts.push(`<b>${escapeHtml(state.video.split(/[\\/]/).pop())}</b>`);
  } else if (state.seq) {
    const tail = state.seq.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
    parts.push(`<b>${escapeHtml(tail)}</b>`);
  }
  if (state.count) parts.push(`${state.count} 帧`);
  if (state.results.length) {
    parts.push(state.results.map((r) => escapeHtml(labelOf(r))).join(" · "));
  }
  if (!parts.length) {
    $("ctxText").textContent = "未选择输入";
    chip.classList.remove("live");
    return;
  }
  $("ctxText").innerHTML = parts.join('<span class="sep">|</span>');
  chip.classList.toggle("live", state.results.length > 0 || (state.kind === "video" && Boolean(state.video)));
}

/* ---------------------------------------------------------------- startup */
async function boot() {
  const data = await apiJson("/api/bootstrap");
  state.boot = data;
  state.runsRoot = data.runs_root;

  for (const mode of data.modes) {
    const opt = document.createElement("option");
    opt.value = mode;
    opt.textContent = data.mode_labels[mode] || mode;
    $("dispMode").appendChild(opt);
  }
  for (const cmap of data.cmaps) {
    const opt = document.createElement("option");
    opt.value = cmap;
    opt.textContent = cmap;
    $("dispCmap").appendChild(opt);
  }
  $("modelPath").value = data.defaults.unet_model || "";

  setOptions($("archiveSelect"), data.archive_tables);
  if (data.archive_tables.length) loadArchive(data.archive_tables[0].path);

  const seqs = await apiJson(`/api/sequences?runs_root=${encodeURIComponent(state.runsRoot)}`);
  setOptions($("seqSelect"), seqs.sequences);
  if (seqs.sequences.length) {
    const preferred = seqs.sequences.find((s) => s.label.includes("结题演示·优先")) || seqs.sequences[0];
    $("seqSelect").value = preferred.path;
    await loadSequence(preferred.path);
  } else {
    $("inputHint").textContent = "runs/ 下未找到 frames.npy";
    updateCtx();
  }
}

/* ---------------------------------------------------------------- input */
async function loadSequence(path) {
  if (!path) return;
  state.kind = "sequence";
  state.seq = path;
  state.index = 0;
  state.quadCache.clear();
  state.videoResult = null;
  $("videoResult").classList.add("hidden");
  stopPlayback();
  $("inputHint").textContent = "正在载入…";

  try {
    const res = await apiJson(
      `/api/results?runs_root=${encodeURIComponent(state.runsRoot)}&sequence_dir=${encodeURIComponent(path)}`
    );
    const auto = res.auto || {};
    const match = auto.predictions
      ? (res.predictions || []).find((p) => p.path === auto.predictions)
      : null;
    state.results = auto.predictions
      ? [{
          method: "existing",
          // Show which run this came from instead of a vague placeholder.
          label: match ? match.label.split("/").slice(-2).join("/") : "已有关联结果",
          detect: auto.predictions,
          track: auto.tracks || null,
          evaluate: auto.metrics || null,
        }]
      : [];
    await refreshMeta();
    $("inputHint").textContent = state.results.length
      ? "已载入该序列的既有检测结果"
      : "该序列暂无检测结果，选择模型后运行即可";
    await renderAll();
  } catch (err) {
    $("inputHint").textContent = err.message;
  }
}

async function refreshMeta() {
  const first = state.results[0];
  const payload = {
    sequence_dir: state.seq,
    detect_dir: first?.detect || null,
    track_dir: first?.track || null,
    evaluate_dir: first?.evaluate || null,
  };
  try {
    state.meta = await apiJson("/api/browser/meta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.count = state.meta.count || 0;
  } catch {
    state.meta = null;
    state.count = 0;
  }
  $("slider").max = Math.max(0, state.count - 1);
  updateCtx();
}

/* ---------------------------------------------------------------- render */
async function renderAll() {
  updateTransport();
  if (state.kind === "video") {
    renderVideoResult();
    return;
  }
  if (!state.results.length) {
    $("emptyState").classList.remove("hidden");
    $("emptyState").textContent = "该序列还没有检测结果。选择模型后点「运行检测」，或换一个已有结果的序列。";
    ["compareGrid", "quadWrap", "metricWrap", "transport"].forEach((id) => $(id).classList.add("hidden"));
    return;
  }
  $("emptyState").classList.add("hidden");
  $("transport").classList.remove("hidden");
  if (state.view === "quad") {
    $("compareGrid").classList.add("hidden");
    $("quadWrap").classList.remove("hidden");
    await renderQuad();
  } else {
    $("quadWrap").classList.add("hidden");
    $("compareGrid").classList.remove("hidden");
    await renderCompare();
  }
  renderTracks();
}

function updateTransport() {
  $("slider").value = state.index;
  $("frameCounter").textContent = state.count ? `${state.index} / ${state.count - 1}` : "0 / 0";
}

async function renderCompare() {
  const sides = state.results.map((r) => ({
    detect_dir: r.detect,
    evaluate_dir: r.evaluate || null,
    label: labelOf(r),
  }));
  let data;
  try {
    data = await apiJson("/api/compare/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: state.seq,
        sides,
        index: state.index,
        settings: getSettings(),
      }),
    });
  } catch (err) {
    $("compareGrid").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
    return;
  }
  state.count = data.count;
  state.index = data.index;
  $("slider").max = Math.max(0, data.count - 1);

  const pixelated = $("dispPixelated").checked;
  $("compareGrid").classList.toggle("pixelated", pixelated);
  $("compareGrid").classList.toggle("single", data.sides.length === 1);
  $("compareGrid").innerHTML = data.sides
    .map((side) => {
      const s = side.summary || {};
      return `
      <figure class="col">
        <div class="col-head"><b>${escapeHtml(side.label)}</b><span class="mono">${data.index + 1}/${data.count}</span></div>
        <div class="col-img"><img src="${side.image}" alt="${escapeHtml(side.label)}" /></div>
        <div class="col-foot">
          <span>IoU <span class="score">${fmt(s.iou)}</span></span>
          <span>虚警/帧 <span class="score">${fmt(s.fp_per_frame, 2)}</span></span>
          <span>重叠 <span class="score">${escapeHtml(side.overlap)}</span> px</span>
        </div>
      </figure>`;
    })
    .join("");

  renderMetricTable(data.sides);
  $("centroidBadge").textContent = data.sides.length > 1
    ? `${data.sides.length} 个模型同屏`
    : `IoU ${fmt(data.sides[0]?.summary?.iou)}`;
  $("cacheBadge").classList.add("hidden");
  updateTransport();
}

function renderMetricTable(sides) {
  if (!sides.some((side) => side.summary)) {
    $("metricWrap").classList.add("hidden");
    return;
  }
  $("metricWrap").classList.remove("hidden");
  const base = sides[0]?.summary || {};
  const head = `<thead><tr><th>指标</th>${sides
    .map((side) => `<th>${escapeHtml(side.label)}</th>`)
    .join("")}</tr></thead>`;

  const body = METRIC_ROWS.map((row) => {
    const cells = sides.map((side, i) => {
      const value = side.summary ? side.summary[row.key] : null;
      let delta = "";
      if (i > 0 && side.summary) {
        const b = base[row.key];
        const v = side.summary[row.key];
        if (Number.isFinite(b) && Number.isFinite(v)) {
          const diff = v - b;
          if (Math.abs(diff) > 1e-9) {
            const good = row.better === "down" ? diff < 0 : diff > 0;
            delta = `<span class="delta ${good ? "up" : "down"}">${good ? "↑" : "↓"}${Math.abs(diff).toFixed(row.digits)}</span>`;
          } else {
            delta = `<span class="delta flat">±0</span>`;
          }
        }
      }
      return `<td class="val">${fmt(value, row.digits)}${delta}</td>`;
    });
    return `<tr><td>${row.label}</td>${cells.join("")}</tr>`;
  }).join("");

  $("metricTable").innerHTML = `${head}<tbody>${body}</tbody>`;
}

function syncQuadSource() {
  const sel = $("quadSource");
  sel.innerHTML = "";
  if (state.results.length < 2) {
    sel.classList.add("hidden");
    return;
  }
  state.results.forEach((r, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = labelOf(r);
    sel.appendChild(opt);
  });
  sel.value = String(state.active);
  sel.classList.remove("hidden");
}

async function renderQuad() {
  syncQuadSource();
  const result = state.results[state.active] || state.results[0];
  if (!result) return;
  const key = `${result.detect}|${state.index}|${settingsKey()}`;
  let data = state.quadCache.get(key);
  if (!data) {
    try {
      data = await apiJson("/api/browser/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence_dir: state.seq,
          detect_dir: result.detect,
          track_dir: result.track || null,
          evaluate_dir: result.evaluate || null,
          index: state.index,
          settings: getSettings(),
        }),
      });
    } catch (err) {
      $("quadWrap").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      return;
    }
    state.quadCache.set(key, data);
    if (state.quadCache.size > 60) state.quadCache.clear();
  }
  for (let i = 0; i < 4; i++) $(`quadImg${i}`).src = data.panels[i];
  state.count = data.count;
  state.index = data.index;
  $("slider").max = Math.max(0, data.count - 1);
  $("centroidBadge").textContent = data.center
    ? `质心 (${data.center[0].toFixed(1)}, ${data.center[1].toFixed(1)})`
    : "本帧无阴影";
  $("quadTag1").textContent = data.index === 0 ? "TEMP-DIFF · 首帧无历史" : "TEMP-DIFF";
  $("cacheBadge").classList.remove("hidden");
  $("cacheBadge").textContent = `缓存 ${state.quadCache.size}`;
  $("quadGrid").classList.toggle("pixelated", $("dispPixelated").checked);
  updateTransport();
}

/* ---------------------------------------------------------------- tracks */
function renderTracks() {
  const rows = state.meta?.track_rows || [];
  if (!rows.length) {
    $("tracksDrawer").classList.add("hidden");
    return;
  }
  $("tracksDrawer").classList.remove("hidden");
  drawLineChart(
    $("coordChart"),
    [
      { name: "x_px", color: "#1a73e8", values: rows.map((r) => r.x_px) },
      { name: "y_px", color: "#f9ab00", values: rows.map((r) => r.y_px) },
    ],
    state.meta.speed_series.frames
  );
  drawLineChart(
    $("speedChart"),
    [{ name: "速度 (px/s)", color: "#1e8e3e", values: state.meta.speed_series.speeds }],
    state.meta.speed_series.frames
  );
  renderTable($("tracksTable"), ["frame", "time_s", "x_px", "y_px", "source", "segment"], rows);
}

/* ---------------------------------------------------------------- charts */
function prepare(canvas, fallbackW, fallbackH) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || fallbackW;
  const h = canvas.clientHeight || fallbackH;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawLineChart(canvas, seriesList, xLabels) {
  const { ctx, w, h } = prepare(canvas, 600, 190);
  const pad = { l: 48, r: 16, t: 28, b: 26 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const all = seriesList.flatMap((s) => s.values.filter((v) => Number.isFinite(v)));
  if (!all.length) {
    ctx.fillStyle = "#747775";
    ctx.font = "13px sans-serif";
    ctx.fillText("暂无轨迹数据", pad.l, pad.t + 32);
    return;
  }
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (hi === lo) { lo -= 1; hi += 1; }

  ctx.strokeStyle = "#e3e8ef";
  ctx.fillStyle = "#747775";
  ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText((hi - ((hi - lo) * i) / 4).toFixed(1), 6, y + 4);
  }

  const n = Math.max(1, xLabels.length - 1);
  seriesList.forEach((s, idx) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) { started = false; return; }
      const x = pad.l + (plotW * i) / n;
      const y = pad.t + plotH * (1 - (v - lo) / (hi - lo));
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(pad.l + idx * 105 + 4, 12, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#444746";
    ctx.font = "500 11.5px sans-serif";
    ctx.fillText(s.name, pad.l + idx * 105 + 13, 16);
  });

  if (xLabels.length) {
    ctx.fillStyle = "#747775";
    ctx.font = "11px monospace";
    const step = Math.max(1, Math.floor(xLabels.length / 6));
    for (let i = 0; i < xLabels.length; i += step) {
      ctx.fillText(String(xLabels[i]), pad.l + (plotW * i) / n - 4, h - 8);
    }
  }
}

function stripSharedPrefix(categories) {
  if (categories.length < 3) return categories;
  const counts = new Map();
  for (const name of categories) {
    const head = String(name).split("_")[0];
    if (!head || head === name || head.length < 2 || head.length > 8) continue;
    counts.set(head, (counts.get(head) || 0) + 1);
  }
  let best = null;
  let bestCount = 0;
  for (const [head, count] of counts) if (count > bestCount) { best = head; bestCount = count; }
  if (!best || bestCount < Math.ceil(categories.length * 0.6)) return categories;
  return categories.map((c) => (String(c).startsWith(best + "_") ? String(c).slice(best.length + 1) : String(c)));
}

function roundedBar(ctx, x, y, w, h, r) {
  if (h <= 0 || w <= 0) return;
  const radius = Math.min(r, w / 2, h);
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
  ctx.fill();
}

function drawBarChart(canvas, chart) {
  const { ctx, w, h } = prepare(canvas, 900, 230);
  if (!chart || !chart.categories.length) {
    ctx.fillStyle = "#747775";
    ctx.font = "13px sans-serif";
    ctx.fillText("该表无可绘制的指标列", 22, 42);
    return;
  }
  const palette = ["#1a73e8", "#1e8e3e", "#f9ab00", "#d93025", "#9334e6", "#12b5cb"];
  const pad = { l: 46, r: 18, t: 38, b: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const maxVal = Math.max(1, ...chart.series.flatMap((s) => s.values));

  ctx.fillStyle = "#1f1f1f";
  ctx.font = "600 12.5px sans-serif";
  ctx.fillText(chart.title || "", pad.l, 20);

  let lx = pad.l + 200;
  ctx.font = "500 11.5px sans-serif";
  chart.series.forEach((s, si) => {
    ctx.fillStyle = palette[si % palette.length];
    ctx.beginPath();
    ctx.arc(lx + 4, 16, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#444746";
    ctx.fillText(s.name, lx + 13, 20);
    lx += Math.min(140, s.name.length * 7.5 + 34);
  });

  ctx.strokeStyle = "#e3e8ef";
  ctx.fillStyle = "#747775";
  ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText((maxVal * (1 - i / 4)).toFixed(2), 6, y + 4);
  }

  const groupW = plotW / chart.categories.length;
  const barW = Math.max(8, Math.min(28, (groupW * 0.7) / Math.max(1, chart.series.length)));
  const labels = stripSharedPrefix(chart.categories);
  chart.categories.forEach((cat, ci) => {
    const gx = pad.l + ci * groupW + (groupW - barW * chart.series.length) / 2;
    chart.series.forEach((s, si) => {
      const v = s.values[ci] || 0;
      const bh = (v / maxVal) * plotH;
      ctx.fillStyle = palette[si % palette.length];
      roundedBar(ctx, gx + si * barW + 1, pad.t + plotH - bh, Math.max(4, barW - 3), bh, 5);
    });
    ctx.fillStyle = "#444746";
    ctx.font = "11.5px sans-serif";
    const raw = labels[ci];
    ctx.fillText(raw.length > 16 ? raw.slice(0, 15) + "…" : raw, pad.l + ci * groupW + 8, h - 12);
  });
}

/* ---------------------------------------------------------------- playback */
function stopPlayback() {
  state.playing = false;
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }
  $("btnPlay").textContent = "▶ 播放";
}

function togglePlayback() {
  if (state.playing) { stopPlayback(); return; }
  if (state.count <= 1) return;
  state.playing = true;
  $("btnPlay").textContent = "⏸ 暂停";
  const step = async () => {
    if (!state.playing) return;
    const t0 = performance.now();
    state.index = (state.index + 1) % state.count;
    try {
      await (state.view === "quad" ? renderQuad() : renderCompare());
    } catch { stopPlayback(); return; }
    const interval = 1000 / Number($("fps").value || 8);
    state.timer = setTimeout(step, Math.max(0, interval - (performance.now() - t0)));
  };
  step();
}

async function gotoFrame(index) {
  stopPlayback();
  state.index = Math.max(0, Math.min(Math.max(0, state.count - 1), index));
  await (state.view === "quad" ? renderQuad() : renderCompare());
  updateTransport();
}

/* ---------------------------------------------------------------- run */
function streamJob(jobId, onDone) {
  const log = $("log");
  log.classList.remove("hidden");
  log.textContent = "";
  setBadge($("runBadge"), "运行中", "warn");
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.addEventListener("log", (ev) => {
    const { line } = JSON.parse(ev.data);
    log.textContent += (log.textContent ? "\n" : "") + line;
    log.scrollTop = log.scrollHeight;
  });
  es.addEventListener("done", (ev) => {
    es.close();
    const { ok, result } = JSON.parse(ev.data);
    setBadge($("runBadge"), ok ? "已完成" : "执行失败", ok ? "ok" : "err");
    onDone(ok, result);
  });
  es.onerror = () => {
    es.close();
    setBadge($("runBadge"), "连接中断", "err");
  };
}

async function runDetection() {
  const methods = [...state.methods];
  if (!methods.length) { alert("请至少选择一个模型"); return; }
  if (state.kind === "video") return runVideoDetection();
  if (!state.seq) { alert("请先选择输入序列"); return; }

  $("btnRun").disabled = true;
  setBadge($("runBadge"), "提交中", "warn");
  try {
    const { job_id } = await apiJson("/api/jobs/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: state.seq,
        methods,
        model: $("modelPath").value.trim() || null,
        threshold: $("advThreshold").value === "" ? null : Number($("advThreshold").value),
        min_area: Number($("advMinArea").value) || 4,
        gate_px: Number($("advGate").value) || 12,
        match_iou: Number($("advIou").value) || 0.5,
      }),
    });
    streamJob(job_id, async (ok, result) => {
      $("btnRun").disabled = false;
      if (!ok || !result) return;
      const done = (result.results || []).filter((r) => r.ok && r.detect);
      if (!done.length) return;
      state.results = done.map((r) => ({
        method: r.method,
        label: r.label,
        detect: r.detect,
        track: r.track,
        evaluate: r.evaluate,
      }));
      state.index = 0;
      state.quadCache.clear();
      state.active = 0;
      await refreshMeta();
      await renderAll();
      $("inputHint").textContent = `已生成 ${done.length} 个模型的结果`;
    });
  } catch (err) {
    $("btnRun").disabled = false;
    setBadge($("runBadge"), "启动失败", "err");
    $("log").classList.remove("hidden");
    $("log").textContent = err.message;
  }
}

async function runVideoDetection() {
  if (!state.video) { alert("请先上传视频或填写路径"); return; }
  const crop = $("useCrop").checked
    ? [Number($("cropX").value), Number($("cropY").value), Number($("cropW").value), Number($("cropH").value)]
    : null;
  $("btnRun").disabled = true;
  setBadge($("runBadge"), "提交中", "warn");
  try {
    const { job_id } = await apiJson("/api/jobs/detect_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_path: state.video,
        crop,
        start_frame: 0,
        max_frames: 300,
        model: $("modelPath").value.trim() || null,
      }),
    });
    streamJob(job_id, (ok, result) => {
      $("btnRun").disabled = false;
      if (!ok || !result) return;
      state.videoResult = result;
      renderVideoResult();
    });
  } catch (err) {
    $("btnRun").disabled = false;
    setBadge($("runBadge"), "启动失败", "err");
    $("log").classList.remove("hidden");
    $("log").textContent = err.message;
  }
}

/* ---------------------------------------------------------------- video */
function showVideoPane() {
  state.kind = "video";
  state.results = [];
  stopPlayback();
  $("seqPane").classList.add("hidden");
  $("videoPane").classList.remove("hidden");
  $("methodChips").classList.add("hidden");
  $("modelField").classList.remove("hidden");
  $("inputHint").textContent = "输入一段雷达视频，逐帧输出检测框与叠加视频";
  $("emptyState").classList.remove("hidden");
  $("emptyState").textContent = "上传视频后先确认首帧与识别区域，再点「运行检测」。";
  ["compareGrid", "quadWrap", "metricWrap", "transport", "videoResult"].forEach((id) => $(id).classList.add("hidden"));
  $("tracksDrawer").classList.add("hidden");
  $("cacheBadge").classList.add("hidden");
  state.count = 0;
  updateTransport();
  updateCtx();
}

function showSequencePane() {
  state.kind = "sequence";
  $("seqPane").classList.remove("hidden");
  $("videoPane").classList.add("hidden");
  $("cropSection").classList.add("hidden");
  $("methodChips").classList.remove("hidden");
  $("modelField").classList.toggle("hidden", !state.methods.has("learned"));
  $("videoResult").classList.add("hidden");
  if (state.seq) loadSequence(state.seq);
  else updateCtx();
}

function loadCropFrame(info) {
  state.video = info.video_path;
  state.videoInfo = info;
  state.crop = { x: 0, y: 0, w: info.width, h: info.height };
  $("dimLabel").textContent = `${info.width} × ${info.height}`;
  $("cropSection").classList.remove("hidden");
  syncCrop();
  const img = new Image();
  img.onload = () => {
    state.cropImg = img;
    drawCrop();
  };
  img.src = `${info.first_frame_url}&t=${Date.now()}`;
  updateCtx();
}

function syncCrop() {
  $("cropX").value = Math.round(state.crop.x);
  $("cropY").value = Math.round(state.crop.y);
  $("cropW").value = Math.round(state.crop.w);
  $("cropH").value = Math.round(state.crop.h);
}

function drawCrop() {
  const canvas = $("cropCanvas");
  const img = state.cropImg;
  if (!img) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const { x, y, w, h } = state.crop;
  ctx.fillStyle = "rgba(0,0,0,0.38)";
  ctx.fillRect(0, 0, canvas.width, y);
  ctx.fillRect(0, y + h, canvas.width, canvas.height - (y + h));
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, canvas.width - (x + w), h);
  ctx.strokeStyle = "#ea4335";
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
  ctx.strokeRect(x, y, w, h);
}

function canvasPoint(ev) {
  const canvas = $("cropCanvas");
  const rect = canvas.getBoundingClientRect();
  const w = state.videoInfo?.width || canvas.width;
  const h = state.videoInfo?.height || canvas.height;
  return {
    x: Math.max(0, Math.min(w, Math.round(((ev.clientX - rect.left) / rect.width) * w))),
    y: Math.max(0, Math.min(h, Math.round(((ev.clientY - rect.top) / rect.height) * h))),
  };
}

function renderVideoResult() {
  const result = state.videoResult;
  $("videoResult").classList.remove("hidden");
  $("emptyState").classList.add("hidden");
  ["compareGrid", "quadWrap", "metricWrap", "transport"].forEach((id) => $(id).classList.add("hidden"));
  if (!result) {
    $("videoOverlay").classList.add("hidden");
    $("videoDownloads").innerHTML = "";
    return;
  }
  $("videoOverlay").classList.remove("hidden");
  if (result.overlay_url) $("videoOverlay").src = `${result.overlay_url}&t=${Date.now()}`;
  const dl = [];
  if (result.overlay_download) dl.push(`<a class="btn btn-sm btn-primary" href="${result.overlay_download}">下载叠加视频</a>`);
  if (result.csv_download) dl.push(`<a class="btn btn-sm" href="${result.csv_download}">下载 predictions.csv</a>`);
  $("videoDownloads").innerHTML = dl.join("");
  renderTable($("videoCsvWrap"), result.csv_fields, result.csv_rows);
}

/* ---------------------------------------------------------------- archive */
async function loadArchive(path) {
  try {
    const data = await apiJson(`/api/archive_table?path=${encodeURIComponent(path)}`);
    drawBarChart($("archiveChart"), data.chart);
    renderTable($("archiveTable"), data.fields, data.rows);
  } catch (err) {
    $("archiveTable").innerHTML = `<div style="padding:16px;color:var(--err);">${escapeHtml(err.message)}</div>`;
  }
}

/* ---------------------------------------------------------------- events */
function bind() {
  $("kindSeg").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-kind]");
    if (!btn) return;
    [...$("kindSeg").children].forEach((b) => b.classList.toggle("on", b === btn));
    if (btn.dataset.kind === "video") showVideoPane();
    else showSequencePane();
  });

  $("seqSelect").addEventListener("change", (e) => loadSequence(e.target.value));
  $("seqPath").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadSequence(e.target.value.trim());
  });

  $("videoFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    $("inputHint").textContent = "正在解码首帧…";
    try {
      const info = await apiJson("/api/video/upload", { method: "POST", body: fd });
      loadCropFrame(info);
      $("inputHint").textContent = "首帧已就绪，确认识别区域后运行检测";
    } catch (err) {
      $("inputHint").textContent = err.message;
    }
  });

  $("btnVideoLoad").addEventListener("click", async () => {
    const path = $("videoPath").value.trim();
    if (!path) return;
    try {
      const info = await apiJson("/api/video/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: path }),
      });
      loadCropFrame(info);
      $("inputHint").textContent = "首帧已就绪，确认识别区域后运行检测";
    } catch (err) {
      $("inputHint").textContent = err.message;
    }
  });

  const canvas = $("cropCanvas");
  canvas.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    state.dragging = true;
    state.dragStart = canvasPoint(ev);
    $("useCrop").checked = true;
  });
  window.addEventListener("mousemove", (ev) => {
    if (!state.dragging || !state.dragStart) return;
    const pt = canvasPoint(ev);
    const x = Math.min(state.dragStart.x, pt.x);
    const y = Math.min(state.dragStart.y, pt.y);
    state.crop = {
      x, y,
      w: Math.max(2, Math.abs(pt.x - state.dragStart.x)),
      h: Math.max(2, Math.abs(pt.y - state.dragStart.y)),
    };
    syncCrop();
    drawCrop();
  });
  window.addEventListener("mouseup", () => { state.dragging = false; });

  ["cropX", "cropY", "cropW", "cropH"].forEach((id) => {
    $(id).addEventListener("input", () => {
      state.crop = {
        x: Math.max(0, Number($("cropX").value) || 0),
        y: Math.max(0, Number($("cropY").value) || 0),
        w: Math.max(2, Number($("cropW").value) || 2),
        h: Math.max(2, Number($("cropH").value) || 2),
      };
      drawCrop();
    });
  });
  $("btnResetCrop").addEventListener("click", () => {
    state.crop = { x: 0, y: 0, w: state.videoInfo?.width || 0, h: state.videoInfo?.height || 0 };
    $("useCrop").checked = false;
    syncCrop();
    drawCrop();
  });

  $("methodChips").addEventListener("click", (ev) => {
    const chip = ev.target.closest("button[data-method]");
    if (!chip) return;
    const method = chip.dataset.method;
    if (state.methods.has(method)) {
      if (state.methods.size === 1) return;
      state.methods.delete(method);
      chip.classList.remove("on");
    } else {
      state.methods.add(method);
      chip.classList.add("on");
    }
    $("modelField").classList.toggle("hidden", !state.methods.has("learned"));
  });

  $("btnRun").addEventListener("click", runDetection);

  $("viewSeg").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-view]");
    if (!btn) return;
    [...$("viewSeg").children].forEach((b) => b.classList.toggle("on", b === btn));
    state.view = btn.dataset.view;
    stopPlayback();
    renderAll();
  });

  $("btnDisplay").addEventListener("click", (ev) => {
    ev.stopPropagation();
    $("displayPop").classList.toggle("hidden");
  });
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest(".pop-wrap")) $("displayPop").classList.add("hidden");
  });
  ["dispMode", "dispCmap", "dispDiff", "dispDenoise", "dispTruth", "dispPred"].forEach((id) => {
    $(id).addEventListener("change", () => {
      state.quadCache.clear();
      if (state.results.length) renderAll();
    });
  });
  $("dispPixelated").addEventListener("change", () => {
    const on = $("dispPixelated").checked;
    $("compareGrid").classList.toggle("pixelated", on);
    $("quadGrid").classList.toggle("pixelated", on);
  });

  $("quadSource").addEventListener("change", (e) => {
    state.active = Number(e.target.value) || 0;
    state.quadCache.clear();
    renderQuad();
  });

  $("btnPlay").addEventListener("click", togglePlayback);
  $("btnPrev").addEventListener("click", () => gotoFrame(state.index - 1));
  $("btnNext").addEventListener("click", () => gotoFrame(state.index + 1));
  $("slider").addEventListener("input", (e) => gotoFrame(Number(e.target.value)));

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (!state.results.length) return;
    if (e.code === "Space") { e.preventDefault(); togglePlayback(); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); gotoFrame(state.index - 1); }
    else if (e.code === "ArrowRight") { e.preventDefault(); gotoFrame(state.index + 1); }
  });

  $("archiveSelect").addEventListener("change", (e) => loadArchive(e.target.value));

  const exportQuad = async (format) => {
    const result = state.results[state.active] || state.results[0];
    if (!result) return;
    $("exportStatus").innerHTML = `<span class="badge warn">正在渲染 ${format.toUpperCase()}…</span>`;
    try {
      const res = await apiJson("/api/browser/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence_dir: state.seq,
          detect_dir: result.detect,
          track_dir: result.track || null,
          evaluate_dir: result.evaluate || null,
          format,
          settings: getSettings(),
        }),
      });
      $("exportStatus").innerHTML =
        `<span class="badge ok">已导出</span> <a class="btn btn-sm btn-primary" href="${res.download_url}">下载 ${escapeHtml(res.filename)}</a>`;
    } catch (err) {
      $("exportStatus").innerHTML = `<span class="badge err">${escapeHtml(err.message)}</span>`;
    }
  };
  $("btnGif").addEventListener("click", () => exportQuad("gif"));
  $("btnMp4").addEventListener("click", () => exportQuad("mp4"));
}

bind();
boot().catch((err) => {
  const el = $("inputHint");
  if (el) el.textContent = `初始化失败：${err.message}`;
});
