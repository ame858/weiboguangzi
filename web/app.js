/* Ka-band Video SAR Workbench Frontend — Material Design 3 (light) edition.
 * Zero external dependencies: no CDN, no framework, works fully offline. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  boot: null,
  runsRoot: "",
  activeTab: "tab1",
  sequences: [],
  // Tab 1
  t1Mode: "modeA",
  bType: "demo",
  t1Browser: { seq: "", detect: "", track: "", eval: "", count: 0, index: 0 },
  t1FrameCache: new Map(),
  t1Inflight: new Set(),
  t1Playing: false,
  t1Timer: null,
  // Tab 2
  t2ExtraPreds: [],
  t2Count: 0,
  t2Index: 0,
  t2Playing: false,
  t2Timer: null,
  // Tab 3
  t3VideoPath: "",
  t3Img: null,
  t3NaturalW: 0,
  t3NaturalH: 0,
  t3Crop: { x: 0, y: 0, w: 0, h: 0 },
  t3Dragging: false,
  t3DragStart: null,
};

function getDisplaySettings() {
  return {
    mode: $("dispMode").value || "percentile",
    cmap: $("dispCmap").value || "gray",
    denoise: $("dispDenoise").checked,
    diff_ref: $("dispDiff").value || "median",
    show_truth: $("dispTruth").checked,
    show_pred: $("dispPred").checked,
  };
}

function settingsKey(s) {
  return `${s.mode}|${s.cmap}|${s.denoise}|${s.diff_ref}|${s.show_truth}|${s.show_pred}`;
}

async function apiJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return data;
}

function showNotice(elId, msg, level = "") {
  const el = $(elId);
  if (!msg) {
    el.classList.add("hidden");
    return;
  }
  el.innerHTML = msg;
  el.className = `notice ${level}`.trim();
  el.classList.remove("hidden");
}

function setOptions(selectEl, items, includeEmpty = false) {
  selectEl.innerHTML = "";
  if (includeEmpty) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "（未选择）";
    selectEl.appendChild(opt);
  }
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = typeof item === "string" ? item : item.path;
    opt.textContent = typeof item === "string" ? item : item.label;
    selectEl.appendChild(opt);
  }
}

function escapeHtml(val) {
  return String(val)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderTable(container, fields, rows) {
  if (!fields || !fields.length || !rows || !rows.length) {
    container.innerHTML = `<div style="padding:16px;color:var(--text-muted);font-size:13px;">暂无数据</div>`;
    return;
  }
  const thead = `<thead><tr>${fields.map((f) => `<th>${escapeHtml(f)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((r) => `<tr>${fields.map((f) => `<td class="mono">${escapeHtml(r[f] ?? "")}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  container.innerHTML = `<table>${thead}${tbody}</table>`;
}

/* =========================================================================
 * SSE Job Streamer
 * ========================================================================= */
function streamJob(jobId, logEl, statusEl, onDone) {
  logEl.textContent = "";
  statusEl.textContent = "运行中...";
  statusEl.className = "badge warn";
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.addEventListener("log", (ev) => {
    const { line } = JSON.parse(ev.data);
    logEl.textContent += (logEl.textContent ? "\n" : "") + line;
    logEl.scrollTop = logEl.scrollHeight;
  });
  es.addEventListener("done", (ev) => {
    es.close();
    const { ok, result } = JSON.parse(ev.data);
    statusEl.textContent = ok ? "已完成" : "执行失败";
    statusEl.className = ok ? "badge ok" : "badge err";
    onDone(ok, result);
  });
  es.onerror = () => {
    es.close();
    statusEl.textContent = "连接中断";
    statusEl.className = "badge err";
  };
}

/* =========================================================================
 * Google-styled Light Canvas Charts (crisp HiDPI, rounded bars)
 * ========================================================================= */
const CHART_INK = "#747775";
const CHART_GRID = "#e3e8ef";
const CHART_LABEL = "#444746";
const PALETTE = ["#1a73e8", "#1e8e3e", "#f9ab00", "#d93025", "#9334e6", "#12b5cb"];

function prepareCanvas(canvas, fallbackW, fallbackH) {
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

function legendDot(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x + 4, y, 4, 0, Math.PI * 2);
  ctx.fill();
}

function drawLineChart(canvas, seriesList, xLabels) {
  const { ctx, w, h } = prepareCanvas(canvas, 560, 195);

  const pad = { l: 48, r: 18, t: 30, b: 28 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const allVals = seriesList.flatMap((s) => s.values.filter((v) => v !== null && Number.isFinite(v)));
  if (!allVals.length) {
    ctx.fillStyle = CHART_INK;
    ctx.font = "13px sans-serif";
    ctx.fillText("暂无有效轨迹数据", pad.l, pad.t + 36);
    return;
  }
  let minY = Math.min(...allVals);
  let maxY = Math.max(...allVals);
  if (maxY === minY) {
    minY -= 1;
    maxY += 1;
  }

  // Soft horizontal gridlines
  ctx.strokeStyle = CHART_GRID;
  ctx.lineWidth = 1;
  ctx.fillStyle = CHART_INK;
  ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    const val = maxY - ((maxY - minY) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, y + 4);
  }

  // Series
  const n = Math.max(1, xLabels.length - 1);
  seriesList.forEach((s, idx) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, i) => {
      if (v === null || !Number.isFinite(v)) {
        started = false;
        return;
      }
      const x = pad.l + (plotW * i) / n;
      const y = pad.t + plotH * (1 - (v - minY) / (maxY - minY));
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Legend
    const lx = pad.l + idx * 110;
    legendDot(ctx, lx, 13, s.color);
    ctx.fillStyle = CHART_LABEL;
    ctx.font = "500 11.5px sans-serif";
    ctx.fillText(s.name, lx + 13, 17);
  });

  // X ticks
  if (xLabels.length) {
    ctx.fillStyle = CHART_INK;
    ctx.font = "11px monospace";
    const step = Math.max(1, Math.floor(xLabels.length / 6));
    for (let i = 0; i < xLabels.length; i += step) {
      const x = pad.l + (plotW * i) / n;
      ctx.fillText(String(xLabels[i]), x - 4, h - 8);
    }
  }
}

function drawRoundedBar(ctx, x, y, w, h, r) {
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

function stripSharedPrefix(categories) {
  if (categories.length < 3) return categories;
  // Pick the most frequent leading token ("unet" in unet_single / unet_temporal…),
  // not the first category's, otherwise a plain name like "threshold" wins.
  const counts = new Map();
  for (const name of categories) {
    const head = String(name).split("_")[0];
    if (!head || head === name || head.length < 2 || head.length > 8) continue;
    counts.set(head, (counts.get(head) || 0) + 1);
  }
  let best = null;
  let bestCount = 0;
  for (const [head, count] of counts) {
    if (count > bestCount) { best = head; bestCount = count; }
  }
  if (!best || bestCount < Math.ceil(categories.length * 0.6)) return categories;
  return categories.map((c) => (String(c).startsWith(best + "_") ? String(c).slice(best.length + 1) : String(c)));
}

function drawGroupedBarChart(canvas, chart) {
  const { ctx, w, h } = prepareCanvas(canvas, 860, 230);

  if (!chart || !chart.categories.length) {
    ctx.fillStyle = CHART_INK;
    ctx.font = "13px sans-serif";
    ctx.fillText("当前归档表无可绘制的柱状指标列", 22, 42);
    return;
  }

  const pad = { l: 46, r: 20, t: 38, b: 38 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const maxVal = Math.max(1.0, ...chart.series.flatMap((s) => s.values));

  ctx.fillStyle = "#1f1f1f";
  ctx.font = "600 12.5px sans-serif";
  ctx.fillText(chart.title || "", pad.l, 20);

  // Legend
  let lx = pad.l + 195;
  ctx.font = "500 11.5px sans-serif";
  chart.series.forEach((s, si) => {
    const color = PALETTE[si % PALETTE.length];
    legendDot(ctx, lx, 16, color);
    ctx.fillStyle = CHART_LABEL;
    ctx.fillText(s.name, lx + 13, 20);
    lx += Math.min(135, s.name.length * 7.5 + 34);
  });

  // Grid
  ctx.strokeStyle = CHART_GRID;
  ctx.fillStyle = CHART_INK;
  ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    const val = maxVal * (1 - i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), 6, y + 4);
  }

  const groupW = plotW / chart.categories.length;
  const barW = Math.max(8, Math.min(28, (groupW * 0.7) / Math.max(1, chart.series.length)));
  // "unet_single / unet_temporal / unet_temporal_noise" all truncate to the same
  // string, so drop a token shared by most categories before shortening.
  const labels = stripSharedPrefix(chart.categories);

  chart.categories.forEach((cat, ci) => {
    const gx = pad.l + ci * groupW + (groupW - barW * chart.series.length) / 2;
    chart.series.forEach((s, si) => {
      const val = s.values[ci] || 0;
      const bh = (val / maxVal) * plotH;
      const x = gx + si * barW;
      const y = pad.t + plotH - bh;
      ctx.fillStyle = PALETTE[si % PALETTE.length];
      drawRoundedBar(ctx, x + 1, y, Math.max(4, barW - 3), bh, 5);
    });
    ctx.fillStyle = CHART_LABEL;
    ctx.font = "11.5px sans-serif";
    const raw = labels[ci];
    const label = raw.length > 16 ? raw.slice(0, 15) + "…" : raw;
    ctx.fillText(label, pad.l + ci * groupW + 8, h - 13);
  });
}

/* =========================================================================
 * Bootstrap & Global Scanning
 * ========================================================================= */
async function bootstrap(runsRoot = "", refresh = false) {
  const query = runsRoot ? `?runs_root=${encodeURIComponent(runsRoot)}` : "";
  const data = await apiJson(`/api/bootstrap${query}`);
  state.boot = data;
  state.runsRoot = data.runs_root;

  $("runsRootInput").value = data.runs_root;
  $("repoRootHint").textContent = `仓库路径：${data.repo_root}`;
  $("runsStatusBadge").textContent = data.runs_exists ? "目录就绪" : "不存在";
  $("runsStatusBadge").className = data.runs_exists ? "badge ok" : "badge warn";

  // Populate display settings once
  if (!$("dispMode").options.length) {
    for (const m of data.modes) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = data.mode_labels[m] || m;
      $("dispMode").appendChild(opt);
    }
    for (const c of data.cmaps) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      $("dispCmap").appendChild(opt);
    }
  }

  $("bDemoConfig").value = data.defaults.demo_config;
  $("bSarModel").value = data.defaults.unet_model;

  renderTab3Static(data);
  setOptions($("t2ArchiveSelect"), data.archive_tables);
  if (data.archive_tables.length) {
    loadArchiveTable(data.archive_tables[0].path);
  }
  await refreshSequences(refresh);
}

async function refreshSequences(refresh = false) {
  const suffix = refresh ? "&refresh=1" : "";
  const data = await apiJson(`/api/sequences?runs_root=${encodeURIComponent(state.runsRoot)}${suffix}`);
  state.sequences = data.sequences;
  $("runsStatusBadge").textContent = data.runs_exists ? `${data.sequences.length} 组序列` : "目录不存在";
  $("runsStatusBadge").className = data.runs_exists && data.sequences.length ? "badge ok" : "badge warn";

  setOptions($("t1SeqSelect"), data.sequences);
  setOptions($("t2SeqSelect"), data.sequences);

  if (data.sequences.length) {
    showNotice("t1Notice", "");
    // The收尾 demo sequence is ranked first by the backend; keep that default.
    const preferred = data.sequences.find((s) => s.label.includes("结题演示·优先")) || data.sequences[0];
    await onTab1SequenceChanged(preferred.path);
    await onTab2SequenceChanged(preferred.path);
  } else {
    showNotice(
      "t1Notice",
      `未在 ${escapeHtml(state.runsRoot)} 找到包含 frames.npy 的目录：可手动填入路径、点击“一键生成快速演示”，或切换到“现场实时运行”。`,
      "warn"
    );
  }
}

/* =========================================================================
 * Tab 1: Simulation & Closeout Browser
 * ========================================================================= */
async function onTab1SequenceChanged(seqDir) {
  if (!seqDir) return;
  try {
    const res = await apiJson(
      `/api/results?runs_root=${encodeURIComponent(state.runsRoot)}&sequence_dir=${encodeURIComponent(seqDir)}`
    );
    const auto = res.auto || {};
    const hasAuto = Boolean(auto.predictions || auto.tracks || auto.metrics);

    // When receipts pin a folder, only that folder is auto-selected; anything
    // unpaired stays "（未选择）" so a mismatched track is never drawn.
    const pick = (selectEl, list, autoPath) => {
      setOptions(selectEl, list, true);
      if (!list.length) return;
      if (list.some((item) => item && item.shape_ok === false)) {
        [...selectEl.options].forEach((opt, index) => {
          if (index === 0) return;
          const item = list[index - 1];
          if (item && item.shape_ok === false) opt.textContent += "（帧形状不符）";
        });
      }
      if (!hasAuto) {
        selectEl.selectedIndex = 1;
        return;
      }
      const idx = list.findIndex((item) => item.path === autoPath);
      selectEl.selectedIndex = idx >= 0 ? idx + 1 : 0;
    };
    pick($("t1DetectSelect"), res.predictions, auto.predictions);
    pick($("t1TrackSelect"), res.tracks, auto.tracks);
    pick($("t1EvalSelect"), res.metrics, auto.metrics);
    $("t1DetectManual").value = "";
    $("t1TrackManual").value = "";
    $("t1EvalManual").value = "";

    const linkedAny = Object.values(res.linked || {}).some(Boolean);
    const paired = hasAuto || linkedAny;
    $("t1LinkBadge").textContent = paired ? "已自动关联" : "全部候选";
    $("t1LinkBadge").className = paired ? "badge ok" : "badge warn";
    await loadTab1BrowserFromControls();
  } catch (err) {
    showNotice("t1Notice", escapeHtml(err.message), "err");
  }
}

async function loadTab1BrowserFromControls() {
  const seq = $("t1SeqManual").value.trim() || $("t1SeqSelect").value;
  const detect = $("t1DetectManual").value.trim() || $("t1DetectSelect").value;
  const track = $("t1TrackManual").value.trim() || $("t1TrackSelect").value;
  const evalDir = $("t1EvalManual").value.trim() || $("t1EvalSelect").value;
  await openTab1Browser({ seq, detect, track, eval: evalDir });
}

async function openTab1Browser(paths) {
  stopTab1Playback();
  if (!paths.seq) return;
  state.t1Browser = { ...paths, count: 0, index: 0 };
  state.t1FrameCache.clear();
  $("t1ActiveSeqLabel").textContent = paths.seq;

  try {
    const meta = await apiJson("/api/browser/meta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: paths.seq,
        detect_dir: paths.detect || null,
        track_dir: paths.track || null,
        evaluate_dir: paths.eval || null,
      }),
    });
    state.t1Browser.count = meta.count;
    $("t1Slider").max = Math.max(0, meta.count - 1);
    $("t1Slider").value = 0;

    if (meta.missing.length) {
      const detail = meta.pred_state === "mismatch"
        ? "当前检测结果的帧形状与本序列不一致"
        : `缺少 ${escapeHtml(meta.missing.join("、"))}`;
      showNotice(
        "t1Notice",
        `${detail}：相关指标显示为 —。可在“现场实时运行”跑完整链路，或展开“结果目录映射”手动指定。`,
        "warn"
      );
    } else {
      showNotice("t1Notice", "");
    }

    // KPI cards
    $("t1DetCards").innerHTML = meta.cards
      .map((c) => `<div class="kpi-card"><div class="kpi-label">${escapeHtml(c.label)}</div><div class="kpi-val">${escapeHtml(c.value)}</div></div>`)
      .join("");
    $("t1TrackCards").innerHTML = meta.tracking
      .map((c) => `<div class="kpi-card"><div class="kpi-label">${escapeHtml(c.label)}</div><div class="kpi-val">${escapeHtml(c.value)}</div></div>`)
      .join("");
    $("t1MetricsMeta").textContent = meta.match_iou
      ? `IoU ≥ ${meta.match_iou} · ${meta.metrics_path || ""}`
      : "";

    // Track charts & table
    if (meta.track_rows && meta.track_rows.length) {
      $("t1TracksPanel").classList.remove("hidden");
      const frames = meta.speed_series.frames;
      drawLineChart(
        $("t1CoordChart"),
        [
          { name: "x_px", color: PALETTE[0], values: meta.track_rows.map((r) => r.x_px) },
          { name: "y_px", color: PALETTE[2], values: meta.track_rows.map((r) => r.y_px) },
        ],
        frames
      );
      drawLineChart(
        $("t1SpeedChart"),
        [{ name: "速度 (px/s)", color: PALETTE[1], values: meta.speed_series.speeds }],
        frames
      );
      $("t1TracksPath").textContent = meta.tracks_path || "";
      renderTable($("t1TracksTableWrap"), ["frame", "time_s", "x_px", "y_px", "source", "segment"], meta.track_rows);
    } else {
      $("t1TracksPanel").classList.add("hidden");
    }

    await renderTab1Frame(0);
    // Warm the cache pool so the first playback pass is already instant.
    prefetchTab1([1, 2, 3, 4]);
  } catch (err) {
    showNotice("t1Notice", escapeHtml(err.message), "err");
  }
}

async function fetchTab1Frame(index) {
  const settings = getDisplaySettings();
  const key = `${index}|${settingsKey(settings)}`;
  if (state.t1FrameCache.has(key)) {
    return state.t1FrameCache.get(key);
  }
  const data = await apiJson("/api/browser/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sequence_dir: state.t1Browser.seq,
      detect_dir: state.t1Browser.detect || null,
      track_dir: state.t1Browser.track || null,
      evaluate_dir: state.t1Browser.eval || null,
      index,
      settings,
    }),
  });
  state.t1FrameCache.set(key, data);
  return data;
}

/* Batch prefetch: one round-trip for several frames, so playback never waits
 * on a cold .npy read. */
async function prefetchTab1(indices) {
  const count = state.t1Browser.count;
  if (!state.t1Browser.seq || count <= 0) return;
  const settings = getDisplaySettings();
  const sKey = settingsKey(settings);
  const need = [];
  for (const i of indices) {
    const idx = ((i % count) + count) % count;
    const key = `${idx}|${sKey}`;
    if (!state.t1FrameCache.has(key) && !state.t1Inflight.has(key)) need.push(idx);
  }
  const unique = [...new Set(need)].slice(0, 8);
  if (!unique.length) return;
  unique.forEach((i) => state.t1Inflight.add(`${i}|${sKey}`));
  try {
    const data = await apiJson("/api/browser/frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: state.t1Browser.seq,
        detect_dir: state.t1Browser.detect || null,
        track_dir: state.t1Browser.track || null,
        evaluate_dir: state.t1Browser.eval || null,
        index: unique[0],
        indices: unique,
        settings,
      }),
    });
    for (const frame of data.frames || []) {
      state.t1FrameCache.set(`${frame.index}|${sKey}`, frame);
    }
  } catch {
    /* prefetch is best-effort */
  } finally {
    unique.forEach((i) => state.t1Inflight.delete(`${i}|${sKey}`));
    updateCacheBadge();
  }
}

function updateCacheBadge() {
  const sKey = settingsKey(getDisplaySettings());
  let hits = 0;
  for (const key of state.t1FrameCache.keys()) {
    if (key.endsWith(`|${sKey}`)) hits += 1;
  }
  $("t1CacheBadge").textContent = `缓存 ${hits}/${state.t1Browser.count}`;
}

async function renderTab1Frame(index) {
  if (!state.t1Browser.seq || state.t1Browser.count <= 0) return;
  const clamped = Math.max(0, Math.min(state.t1Browser.count - 1, index));
  state.t1Browser.index = clamped;
  $("t1Slider").value = clamped;
  $("t1FrameCounter").textContent = `${clamped} / ${state.t1Browser.count - 1}`;

  const data = await fetchTab1Frame(clamped);
  for (let i = 0; i < 4; i++) {
    $(`quadImg${i}`).src = data.panels[i];
  }
  $("t1CentroidBadge").textContent = data.center
    ? `质心 (${data.center[0].toFixed(1)}, ${data.center[1].toFixed(1)})`
    : "本帧无阴影";
  // Frame 0 has no history, so the difference view is legitimately flat.
  $("quadTag1").textContent = clamped === 0 && $("dispDiff").value === "median"
    ? "TEMP-DIFF · 首帧无历史"
    : "TEMP-DIFF";
  updateCacheBadge();

  // Drop stale settings variants so the pool cannot grow without bound.
  if (state.t1FrameCache.size > Math.max(24, state.t1Browser.count * 3)) {
    const sKey = settingsKey(getDisplaySettings());
    for (const key of [...state.t1FrameCache.keys()]) {
      if (!key.endsWith(`|${sKey}`)) state.t1FrameCache.delete(key);
    }
  }
}

function stopTab1Playback() {
  state.t1Playing = false;
  if (state.t1Timer) {
    clearTimeout(state.t1Timer);
    state.t1Timer = null;
  }
  $("t1PlayBtn").textContent = "▶ 播放";
}

function toggleTab1Playback() {
  if (state.t1Playing) {
    stopTab1Playback();
    return;
  }
  if (state.t1Browser.count <= 1) return;
  state.t1Playing = true;
  $("t1PlayBtn").textContent = "⏸ 暂停";

  const step = async () => {
    if (!state.t1Playing) return;
    const t0 = performance.now();
    const nextIdx = (state.t1Browser.index + 1) % state.t1Browser.count;
    try {
      await renderTab1Frame(nextIdx);
    } catch {
      stopTab1Playback();
      return;
    }
    prefetchTab1([nextIdx + 1, nextIdx + 2, nextIdx + 3, nextIdx + 4]);
    const interval = 1000 / Number($("t1FpsSelect").value || 8);
    state.t1Timer = setTimeout(step, Math.max(0, interval - (performance.now() - t0)));
  };
  step();
}

async function exportTab1(format) {
  if (!state.t1Browser.seq) return;
  $("t1ExportStatus").innerHTML = `<span class="badge warn">正在渲染 ${format.toUpperCase()} ...</span>`;
  try {
    const res = await apiJson("/api/browser/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: state.t1Browser.seq,
        detect_dir: state.t1Browser.detect || null,
        track_dir: state.t1Browser.track || null,
        evaluate_dir: state.t1Browser.eval || null,
        format,
        settings: getDisplaySettings(),
      }),
    });
    $("t1ExportStatus").innerHTML = `
      <span class="badge ok">导出成功</span>
      <span class="mono section-sub" style="margin:0 8px;">${escapeHtml(res.path)}</span>
      <a class="btn btn-sm btn-primary" href="${res.download_url}">下载 ${escapeHtml(res.filename)}</a>
    `;
  } catch (err) {
    $("t1ExportStatus").innerHTML = `<span class="badge err">${escapeHtml(err.message)}</span>`;
  }
}

async function runLiveJob(endpoint, payload) {
  $("t1LiveLogWrap").classList.remove("hidden");
  try {
    const { job_id } = await apiJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    streamJob(job_id, $("t1LiveLog"), $("t1JobStatus"), async (ok, result) => {
      if (!(ok && result)) return;
      // Persist the live result straight into the browser without losing it.
      for (const id of ["t1DetectManual", "t1TrackManual", "t1EvalManual"]) $(id).value = "";
      await refreshSequencesSilently();
      await openTab1Browser({
        seq: result.sequence,
        detect: result.predictions,
        track: result.tracks,
        eval: result.metrics,
      });
    });
  } catch (err) {
    $("t1JobStatus").textContent = "启动失败";
    $("t1JobStatus").className = "badge err";
    $("t1LiveLog").textContent = err.message;
  }
}

async function refreshSequencesSilently() {
  try {
    const data = await apiJson(`/api/sequences?runs_root=${encodeURIComponent(state.runsRoot)}&refresh=1`);
    state.sequences = data.sequences;
    setOptions($("t1SeqSelect"), data.sequences);
    setOptions($("t2SeqSelect"), data.sequences);
  } catch {
    /* keep current lists */
  }
}

/* =========================================================================
 * Tab 2: Side-by-side Algorithm Comparison & Archive Charts
 * ========================================================================= */
async function onTab2SequenceChanged(seqDir) {
  if (!seqDir) return;
  try {
    const res = await apiJson(
      `/api/results?runs_root=${encodeURIComponent(state.runsRoot)}&sequence_dir=${encodeURIComponent(seqDir)}`
    );
    const paths = [
      ...state.t2ExtraPreds,
      ...res.predictions.filter((p) => p.shape_ok !== false).map((p) => p.path),
    ];
    const unique = [...new Set(paths)];
    setOptions($("t2LeftSelect"), unique);
    setOptions($("t2RightSelect"), unique);
    if (unique.length > 1) {
      const left = unique[0];
      const other = unique.findIndex((p) => p !== left);
      if (other > 0) $("t2RightSelect").selectedIndex = other;
    }
    state.t2Index = 0;
    $("t2Slider").max = 0;
    if (!unique.length) return;
    const meta = await apiJson("/api/browser/meta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence_dir: seqDir }),
    });
    state.t2Count = meta.count;
    $("t2Slider").max = Math.max(0, meta.count - 1);
    await renderTab2Frame(0);
  } catch (err) {
    $("t2LeftMeta").textContent = err.message;
  }
}

async function renderTab2Frame(index) {
  const seq = $("t2SeqSelect").value;
  const left = $("t2LeftSelect").value;
  const right = $("t2RightSelect").value;
  if (!seq || !left || !right) return;
  const data = await apiJson("/api/compare/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sequence_dir: seq,
      left_dir: left,
      right_dir: right,
      index,
      settings: getDisplaySettings(),
    }),
  });
  state.t2Count = data.count;
  state.t2Index = data.index;
  $("t2Slider").max = Math.max(0, data.count - 1);
  $("t2Slider").value = data.index;
  $("t2FrameCounter").textContent = `${data.index} / ${data.count - 1}`;

  $("t2LeftImg").src = data.sides[0].image;
  $("t2LeftMeta").textContent = `连通域 ${data.sides[0].components} · 重叠 ${data.sides[0].overlap} px`;
  $("t2RightImg").src = data.sides[1].image;
  $("t2RightMeta").textContent = `连通域 ${data.sides[1].components} · 重叠 ${data.sides[1].overlap} px`;
}

function toggleTab2Playback() {
  if (state.t2Playing) {
    state.t2Playing = false;
    clearTimeout(state.t2Timer);
    $("t2PlayBtn").textContent = "▶ 播放";
    return;
  }
  if (state.t2Count <= 1) return;
  state.t2Playing = true;
  $("t2PlayBtn").textContent = "⏸ 暂停";
  const step = async () => {
    if (!state.t2Playing) return;
    try {
      await renderTab2Frame((state.t2Index + 1) % state.t2Count);
    } catch {
      state.t2Playing = false;
      $("t2PlayBtn").textContent = "▶ 播放";
      return;
    }
    state.t2Timer = setTimeout(step, 150);
  };
  step();
}

async function loadArchiveTable(path) {
  try {
    const data = await apiJson(`/api/archive_table?path=${encodeURIComponent(path)}`);
    drawGroupedBarChart($("t2ArchiveChart"), data.chart);
    renderTable($("t2ArchiveTableWrap"), data.fields, data.rows);
  } catch (err) {
    $("t2ArchiveTableWrap").innerHTML =
      `<div style="padding:16px;color:var(--err-text);font-size:13px;">${escapeHtml(err.message)}</div>`;
  }
}

/* =========================================================================
 * Tab 3: Real Radar Video Inference & Interactive Canvas Cropper
 * ========================================================================= */
function renderTab3Static(boot) {
  const env = boot.environment;
  const badges = [
    { name: "FFmpeg", ok: Boolean(env.ffmpeg) },
    { name: "PyTorch", ok: Boolean(env.torch) },
    { name: "torchvision", ok: Boolean(env.torchvision) },
    { name: "imageio-ffmpeg", ok: Boolean(env.imageio_ffmpeg) },
  ];
  $("t3EnvBadges").innerHTML = badges
    .map((b) => `<span class="badge ${b.ok ? "ok" : "err"}">${b.ok ? "✓" : "✕"} ${b.name}</span>`)
    .join("");

  const hints = [];
  if (!env.ffmpeg) hints.push(`缺少 FFmpeg：<code>${escapeHtml(boot.defaults.ffmpeg_install)}</code>`);
  if (!env.torch) hints.push(`缺少 PyTorch：<code>${escapeHtml(boot.defaults.torch_install)}</code>`);
  hints.push(`解释器：<span class="mono">${escapeHtml(env.python)}</span>`);
  $("t3EnvHints").innerHTML = hints.join("<br>");

  const info = boot.snl_model_info || {};
  const dev = info.development || {};
  const devPairs = [
    ["开发复核帧数", dev.frames],
    ["命中 TP", dev.tp],
    ["误报 FP", dev.fp],
    ["漏检 FN", dev.fn],
    ["查准率", dev.precision],
    ["召回率", dev.recall],
    ["F1", dev.f1],
    ["每帧虚警", dev.fp_per_frame],
  ];
  $("t3ModelCards").innerHTML = devPairs
    .map(([k, v]) => `<div class="kpi-card"><div class="kpi-label">${k}</div><div class="kpi-val">${v == null ? "—" : typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v}</div></div>`)
    .join("");
  $("t3ModelMeta").textContent = info.architecture
    ? `架构：${info.architecture} · 通道：${info.input_channels} · 阈值：${info.threshold} · Epoch：${info.epoch}`
    : "未读取到 stacked-center.json";

  setOptions($("t3PreviewSelect"), boot.preview_videos);
  if (boot.preview_videos.length) {
    $("t3PreviewVideo").src = `/api/media?path=${encodeURIComponent(boot.preview_videos[0])}`;
  } else {
    $("t3PreviewWrap").innerHTML = `<div style="padding:28px 16px;text-align:center;color:var(--text-muted);font-size:13px;background:var(--surface-tonal);border-radius:16px;">未找到归档演示视频，可在下方运行 detect-video 生成。</div>`;
  }
}

function loadCropFirstFrame(info) {
  state.t3VideoPath = info.video_path;
  state.t3NaturalW = info.width;
  state.t3NaturalH = info.height;
  state.t3Crop = { x: 0, y: 0, w: info.width, h: info.height };
  syncCropInputs();
  $("t3DimLabel").textContent = `${info.width} × ${info.height}`;
  $("t3CropSection").classList.remove("hidden");

  const img = new Image();
  img.onload = () => {
    state.t3Img = img;
    drawCropCanvas();
  };
  img.onerror = () => {
    $("t3DimLabel").textContent = "首帧加载失败";
  };
  img.src = `${info.first_frame_url}&t=${Date.now()}`;
}

function syncCropInputs() {
  $("t3CropX").value = Math.round(state.t3Crop.x);
  $("t3CropY").value = Math.round(state.t3Crop.y);
  $("t3CropW").value = Math.round(state.t3Crop.w);
  $("t3CropH").value = Math.round(state.t3Crop.h);
}

function drawCropCanvas() {
  const canvas = $("t3CropCanvas");
  const img = state.t3Img;
  if (!img) return;
  canvas.width = state.t3NaturalW;
  canvas.height = state.t3NaturalH;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  const { x, y, w, h } = state.t3Crop;
  ctx.fillStyle = "rgba(0, 0, 0, 0.38)";
  ctx.fillRect(0, 0, canvas.width, y);
  ctx.fillRect(0, y + h, canvas.width, canvas.height - (y + h));
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, canvas.width - (x + w), h);

  ctx.strokeStyle = "#ea4335";
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
  ctx.strokeRect(x, y, w, h);
}

function canvasPoint(ev) {
  const canvas = $("t3CropCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(state.t3NaturalW, Math.round(((ev.clientX - rect.left) / rect.width) * state.t3NaturalW))),
    y: Math.max(0, Math.min(state.t3NaturalH, Math.round(((ev.clientY - rect.top) / rect.height) * state.t3NaturalH))),
  };
}

/* =========================================================================
 * Event Wiring
 * ========================================================================= */
function bindEvents() {
  // Top tab navigation
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      stopTab1Playback();
      state.activeTab = btn.dataset.tab;
      document.querySelectorAll(".nav-tab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-view").forEach((v) => v.classList.toggle("hidden", v.id !== state.activeTab));
      if (state.activeTab === "tab1") updateCacheBadge();
      if (state.activeTab === "tab2") {
        // Bootstrap can still be finishing when the user switches early.
        if (!$("t2LeftSelect").options.length && $("t2SeqSelect").value) {
          onTab2SequenceChanged($("t2SeqSelect").value);
        } else if ($("t2ArchiveSelect").value) {
          loadArchiveTable($("t2ArchiveSelect").value);
        }
      }
    });
  });

  // Global runs root
  $("btnReloadRuns").addEventListener("click", () => bootstrap($("runsRootInput").value.trim(), true));
  $("runsRootInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") bootstrap($("runsRootInput").value.trim(), true);
  });

  // Display settings (instant visual refresh, never clears selection)
  ["dispMode", "dispCmap", "dispDiff", "dispDenoise", "dispTruth", "dispPred"].forEach((id) => {
    $(id).addEventListener("change", () => {
      state.t1FrameCache.clear();
      if (state.activeTab === "tab1") renderTab1Frame(state.t1Browser.index);
      if (state.activeTab === "tab2") renderTab2Frame(state.t2Index);
    });
  });
  $("dispPixelated").addEventListener("change", () => {
    const on = $("dispPixelated").checked;
    $("t1QuadGrid").classList.toggle("pixelated", on);
    $("t2Grid").classList.toggle("pixelated", on);
  });

  // Tab 1 mode switch
  document.querySelectorAll("[data-t1mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.t1Mode = btn.dataset.t1mode;
      document.querySelectorAll("[data-t1mode]").forEach((b) => b.classList.toggle("active", b === btn));
      $("t1ModeAPanel").classList.toggle("hidden", state.t1Mode !== "modeA");
      $("t1ModeBPanel").classList.toggle("hidden", state.t1Mode !== "modeB");
    });
  });

  // Tab 1 selectors
  $("t1SeqSelect").addEventListener("change", () => {
    $("t1SeqManual").value = "";
    onTab1SequenceChanged($("t1SeqSelect").value);
  });
  $("t1SeqManual").addEventListener("keydown", (e) => {
    if (e.key === "Enter") onTab1SequenceChanged($("t1SeqManual").value.trim());
  });
  ["t1DetectSelect", "t1TrackSelect", "t1EvalSelect"].forEach((id) =>
    $(id).addEventListener("change", loadTab1BrowserFromControls)
  );
  ["t1DetectManual", "t1TrackManual", "t1EvalManual"].forEach((id) =>
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadTab1BrowserFromControls();
    })
  );

  // Tab 1 playback & export
  $("t1Slider").addEventListener("input", (e) => {
    stopTab1Playback();
    renderTab1Frame(Number(e.target.value));
  });
  $("t1PlayBtn").addEventListener("click", toggleTab1Playback);
  $("t1PrevBtn").addEventListener("click", () => {
    stopTab1Playback();
    renderTab1Frame(state.t1Browser.index - 1);
  });
  $("t1NextBtn").addEventListener("click", () => {
    stopTab1Playback();
    renderTab1Frame(state.t1Browser.index + 1);
  });
  $("t1ExportGif").addEventListener("click", () => exportTab1("gif"));
  $("t1ExportMp4").addEventListener("click", () => exportTab1("mp4"));

  // Keyboard shortcuts (Tab 1)
  window.addEventListener("keydown", (e) => {
    if (state.activeTab !== "tab1") return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      toggleTab1Playback();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      stopTab1Playback();
      renderTab1Frame(state.t1Browser.index - 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      stopTab1Playback();
      renderTab1Frame(state.t1Browser.index + 1);
    }
  });

  // Tab 1 mode B
  document.querySelectorAll("[data-btype]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.bType = btn.dataset.btype;
      document.querySelectorAll("[data-btype]").forEach((b) => b.classList.toggle("active", b === btn));
      $("bDemoForm").classList.toggle("hidden", state.bType !== "demo");
      $("bSarForm").classList.toggle("hidden", state.bType !== "sar");
    });
  });
  $("bDemoFrames").addEventListener("input", (e) => ($("bDemoFramesVal").textContent = e.target.value));
  $("bSarMethod").addEventListener("change", () => {
    $("bSarModelWrap").style.opacity = $("bSarMethod").value === "learned" ? "1" : "0.45";
  });
  $("bSarModelWrap").style.opacity = "0.45";

  $("btnRunDemo").addEventListener("click", () =>
    runLiveJob("/api/jobs/demo", {
      config: $("bDemoConfig").value.trim(),
      frames: Number($("bDemoFrames").value),
    })
  );
  $("btnBootstrapDemo").addEventListener("click", () => {
    document.querySelector('[data-t1mode="modeB"]').click();
    runLiveJob("/api/jobs/demo", {
      config: $("bDemoConfig").value.trim(),
      frames: 16,
    });
  });
  $("btnRunSar").addEventListener("click", () =>
    runLiveJob("/api/jobs/sar", {
      bandwidth_hz: Number($("bSarBw").value),
      method: $("bSarMethod").value,
      model: $("bSarModel").value.trim(),
    })
  );

  // Tab 2
  $("t2SeqSelect").addEventListener("change", () => onTab2SequenceChanged($("t2SeqSelect").value));
  $("t2LeftSelect").addEventListener("change", () => renderTab2Frame(state.t2Index));
  $("t2RightSelect").addEventListener("change", () => renderTab2Frame(state.t2Index));
  $("t2Slider").addEventListener("input", (e) => renderTab2Frame(Number(e.target.value)));
  $("t2PlayBtn").addEventListener("click", toggleTab2Playback);
  $("t2ArchiveSelect").addEventListener("change", (e) => loadArchiveTable(e.target.value));
  $("t2CompareRunBtn").addEventListener("click", async () => {
    const seq = $("t2SeqSelect").value;
    if (!seq) return;
    $("t2LogWrap").classList.remove("hidden");
    try {
      const { job_id } = await apiJson("/api/jobs/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence_dir: seq }),
      });
      streamJob(job_id, $("t2LiveLog"), $("t2JobStatus"), async (ok, result) => {
        if (ok && result?.predictions) {
          state.t2ExtraPreds = [...result.predictions, ...state.t2ExtraPreds];
          await onTab2SequenceChanged(seq);
        }
      });
    } catch (err) {
      $("t2JobStatus").textContent = "启动失败";
      $("t2JobStatus").className = "badge err";
      $("t2LiveLog").textContent = err.message;
    }
  });

  // Tab 3
  $("t3PreviewSelect").addEventListener("change", (e) => {
    $("t3PreviewVideo").src = `/api/media?path=${encodeURIComponent(e.target.value)}`;
  });

  $("t3FileInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    $("t3DimLabel").textContent = "解码首帧…";
    try {
      const info = await apiJson("/api/video/upload", { method: "POST", body: fd });
      loadCropFirstFrame(info);
    } catch (err) {
      alert(err.message);
    }
  });

  $("t3LoadLocalBtn").addEventListener("click", async () => {
    const p = $("t3LocalPath").value.trim();
    if (!p) return;
    try {
      const info = await apiJson("/api/video/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: p }),
      });
      loadCropFirstFrame(info);
    } catch (err) {
      alert(err.message);
    }
  });

  // Interactive canvas crop
  const cropCanvas = $("t3CropCanvas");
  cropCanvas.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    state.t3Dragging = true;
    state.t3DragStart = canvasPoint(ev);
    $("t3UseCrop").checked = true;
  });
  window.addEventListener("mousemove", (ev) => {
    if (!state.t3Dragging || !state.t3DragStart) return;
    const pt = canvasPoint(ev);
    const x = Math.min(state.t3DragStart.x, pt.x);
    const y = Math.min(state.t3DragStart.y, pt.y);
    const w = Math.max(2, Math.abs(pt.x - state.t3DragStart.x));
    const h = Math.max(2, Math.abs(pt.y - state.t3DragStart.y));
    state.t3Crop = {
      x,
      y,
      w: Math.min(w, state.t3NaturalW - x),
      h: Math.min(h, state.t3NaturalH - y),
    };
    syncCropInputs();
    drawCropCanvas();
  });
  window.addEventListener("mouseup", () => {
    state.t3Dragging = false;
  });

  ["t3CropX", "t3CropY", "t3CropW", "t3CropH"].forEach((id) => {
    $(id).addEventListener("input", () => {
      state.t3Crop = {
        x: Math.max(0, Number($("t3CropX").value) || 0),
        y: Math.max(0, Number($("t3CropY").value) || 0),
        w: Math.max(2, Number($("t3CropW").value) || 2),
        h: Math.max(2, Number($("t3CropH").value) || 2),
      };
      drawCropCanvas();
    });
  });

  $("t3ResetCropBtn").addEventListener("click", () => {
    state.t3Crop = { x: 0, y: 0, w: state.t3NaturalW, h: state.t3NaturalH };
    $("t3UseCrop").checked = false;
    syncCropInputs();
    drawCropCanvas();
  });

  $("t3RunBtn").addEventListener("click", async () => {
    const vpath = state.t3VideoPath || $("t3LocalPath").value.trim();
    if (!vpath) {
      alert("请先上传视频或填写本地视频路径");
      return;
    }
    const crop = $("t3UseCrop").checked
      ? [
          Number($("t3CropX").value),
          Number($("t3CropY").value),
          Number($("t3CropW").value),
          Number($("t3CropH").value),
        ]
      : null;
    $("t3LogWrap").classList.remove("hidden");
    try {
      const { job_id } = await apiJson("/api/jobs/detect_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: vpath,
          crop,
          start_frame: Number($("t3StartFrame").value) || 0,
          max_frames: Number($("t3MaxFrames").value) || 24,
          model: $("t3ModelPath").value.trim() || null,
        }),
      });
      streamJob(job_id, $("t3LiveLog"), $("t3JobStatus"), (ok, result) => {
        if (!ok || !result) return;
        $("t3ResultWrap").classList.remove("hidden");
        const dl = [];
        if (result.overlay_download) dl.push(`<a class="btn btn-sm btn-primary" href="${result.overlay_download}">下载 overlay.mp4</a>`);
        if (result.csv_download) dl.push(`<a class="btn btn-sm" href="${result.csv_download}">下载 predictions.csv</a>`);
        $("t3Downloads").innerHTML = dl.join("");
        if (result.overlay_url) $("t3ResultVideo").src = `${result.overlay_url}&t=${Date.now()}`;
        renderTable($("t3CsvTableWrap"), result.csv_fields, result.csv_rows);
        $("t3VideoJson").textContent = JSON.stringify(result.video_json || {}, null, 2);
      });
    } catch (err) {
      $("t3JobStatus").textContent = "启动失败";
      $("t3JobStatus").className = "badge err";
      $("t3LiveLog").textContent = err.message;
    }
  });
}

bindEvents();
bootstrap().catch((err) => showNotice("t1Notice", `初始化失败：${escapeHtml(err.message)}`, "err"));
