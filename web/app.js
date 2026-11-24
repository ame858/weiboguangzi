/* Ka-band video SAR shadow comparison workbench.
- Layout: fixed rail (input / model / run) + canvas (columns / pills / filmstrip).
- No framework, no CDN, fully offline. */
"use strict";

const $ = (id) => document.getElementById(id);

const ICON_PLAY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
const ICON_PAUSE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
const ICON_CHEVRON = '<svg class="m3-select-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"></polyline></svg>';
const ICON_CHECK = '<svg class="m3-menu-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>';

function setPlayState(playing) {
  const icon = $("playIcon");
  const label = $("playLabel");
  const btn = $("btnPlay");
  if (!icon || !label) return;
  icon.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
  label.textContent = playing ? "暂停" : "播放";
  if (btn) btn.classList.toggle("playing", Boolean(playing));
}

const METHODS = {
  threshold: "局部暗度阈值",
  difference: "时序背景差分",
  learned: "U-Net 时序",
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
  methods: new Set(["threshold", "difference", "learned"]),
  results: [],
  slots: [],
  quadSlot: 0,
  meta: null,
  count: 0,
  index: 0,
  view: "compare",
  playing: false,
  timer: null,
  quadCache: new Map(),
  strip: null,
  stripKey: "",
  sides: [],
  videoResult: null,
  stepNo: 0,
};

/* ------------------------------------------------------------ Gemini M3 custom select */
let activeM3Menu = null;
let activeM3Trigger = null;

function closeM3Menu(restoreFocus = false) {
  if (activeM3Menu) {
    activeM3Menu.remove();
    activeM3Menu = null;
  }
  if (activeM3Trigger) {
    activeM3Trigger.setAttribute("aria-expanded", "false");
    if (restoreFocus) activeM3Trigger.focus();
    activeM3Trigger = null;
  }
}

function syncM3Select(sel) {
  if (!sel || !sel._m3Wrap) return;
  const wrap = sel._m3Wrap;
  const label = sel._m3Label;
  wrap.classList.toggle("hidden", sel.classList.contains("hidden"));
  const opt = sel.options[sel.selectedIndex] || sel.options[0];
  label.textContent = opt ? opt.textContent : "—";
  if (sel._m3Trigger) {
    sel._m3Trigger.title = opt ? (opt.title || opt.textContent) : "";
    sel._m3Trigger.disabled = sel.disabled;
  }
}

function openM3Menu(sel) {
  const trigger = sel._m3Trigger;
  if (!trigger) return;
  const isSame = activeM3Trigger === trigger;
  closeM3Menu(false);
  if (isSame) return;

  const menu = document.createElement("div");
  menu.className = "m3-menu";
  menu.setAttribute("role", "listbox");

  [...sel.options].forEach((opt) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "m3-menu-item";
    item.setAttribute("role", "option");
    const selected = opt.value === sel.value;
    if (selected) {
      item.classList.add("is-selected");
      item.setAttribute("aria-selected", "true");
    }
    item.innerHTML = `<span class="m3-menu-item-text">${esc(opt.textContent)}</span>${selected ? ICON_CHECK : ""}`;
    if (opt.title) item.title = opt.title;
    item.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const changed = sel.value !== opt.value;
      sel.value = opt.value;
      syncM3Select(sel);
      closeM3Menu(true);
      if (changed) sel.dispatchEvent(new Event("change", { bubbles: true }));
    });
    menu.appendChild(item);
  });

  document.body.appendChild(menu);
  activeM3Menu = menu;
  activeM3Trigger = trigger;
  trigger.setAttribute("aria-expanded", "true");

  // 视口自适应定位：贴紧触发器，靠近底部自动向上翻，靠近右侧自动右对齐
  const rect = trigger.getBoundingClientRect();
  const minW = Math.max(rect.width, 148);
  menu.style.minWidth = `${Math.min(minW, 300)}px`;
  const mRect = menu.getBoundingClientRect();
  const pad = 10;

  let left = rect.left;
  if (left + mRect.width > window.innerWidth - pad) {
    left = Math.max(pad, rect.right - mRect.width);
  }
  let top = rect.bottom + 6;
  if (top + mRect.height > window.innerHeight - pad && rect.top - 6 - mRect.height > pad) {
    top = rect.top - 6 - mRect.height;
    menu.classList.add("flip-up");
  }
  menu.style.left = `${Math.round(left)}px`;
  menu.style.top = `${Math.round(top)}px`;

  const selectedItem = menu.querySelector(".is-selected");
  if (selectedItem) selectedItem.scrollIntoView({ block: "nearest" });
}

function upgradeSelect(sel) {
  if (!sel || sel._m3Wrap) {
    if (sel) syncM3Select(sel);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "m3-select";
  if (sel.style.maxWidth) wrap.style.maxWidth = sel.style.maxWidth;

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "m3-select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  if (sel.getAttribute("aria-label")) {
    trigger.setAttribute("aria-label", sel.getAttribute("aria-label"));
  }

  const label = document.createElement("span");
  label.className = "m3-select-label";
  trigger.appendChild(label);
  trigger.insertAdjacentHTML("beforeend", ICON_CHEVRON);

  sel.parentNode.insertBefore(wrap, sel);
  wrap.appendChild(sel);
  wrap.appendChild(trigger);
  sel.classList.add("native-select-hidden");

  sel._m3Wrap = wrap;
  sel._m3Trigger = trigger;
  sel._m3Label = label;

  trigger.addEventListener("click", (ev) => {
    ev.stopPropagation();
    openM3Menu(sel);
  });

  sel.addEventListener("change", () => syncM3Select(sel));

  // 自动监听 option 变化和 hidden 类名变化
  const obs = new MutationObserver(() => syncM3Select(sel));
  obs.observe(sel, { childList: true, attributes: true, attributeFilter: ["class", "disabled"] });

  syncM3Select(sel);
}

function upgradeAllSelects(root = document) {
  root.querySelectorAll("select").forEach((sel) => {
    // 空槽位内部隐藏的占位 select 不需要渲染触发器
    if (sel.closest(".col-empty")) return;
    upgradeSelect(sel);
  });
}

/* ------------------------------------------------------------ utilities */
async function apiJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

const esc = (v) =>
  String(v).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const fmt = (v, digits = 3) =>
  v === null || v === undefined || !Number.isFinite(v)
    ? "—"
    : digits === 0 ? String(Math.round(v)) : v.toFixed(digits);

function padSlots(list) {
  const out = list.slice(0, 3);
  while (out.length < 3) out.push(-1);
  return out;
}

function labelOf(result) {
  if (!result) return "—";
  return result.method && METHODS[result.method] ? METHODS[result.method] : result.label || "已有结果";
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

function setBadge(el, text, level = "") {
  el.textContent = text;
  el.className = `badge ${level}`.trim();
}

function closeSlotMenus() {
  document.querySelectorAll("#columns .slot-menu").forEach((m) => m.remove());
  document.querySelectorAll("#columns .col-empty.menu-open").forEach((c) => c.classList.remove("menu-open"));
}

function selectSlotValue(slot, value) {
  if (value >= 0) {
    // Picking a result that already occupies another column swaps the two,
    // so a comparison never shows the same model twice.
    const other = state.slots.indexOf(value);
    if (other >= 0 && other !== slot) state.slots[other] = state.slots[slot];
  }
  state.slots[slot] = value;
  render();
}

/* ------------------------------------------------------------ context */
function updateCtx() {
  const parts = [];
  if (state.kind === "video") {
    if (state.video) parts.push(`<b>${esc(state.video.split(/[\\/]/).pop())}</b>`);
  } else if (state.seq) {
    const tail = state.seq.split(/[\\/]/).filter(Boolean).slice(-1)[0];
    parts.push(`<b>${esc(tail)}</b>`);
  }
  if (state.count) parts.push(`${state.count} 帧`);
  const shown = state.slots.map((i) => labelOf(state.results[i])).filter((x) => x !== "—");
  if (shown.length) {
    parts.push(shown.length > 2
      ? `${shown.slice(0, 2).map(esc).join(" · ")} 等 ${shown.length} 个`
      : shown.map(esc).join(" · "));
  }
  if (!parts.length) {
    $("ctxText").textContent = "未选择输入";
    $("ctxChip").classList.remove("live");
    return;
  }
  $("ctxText").innerHTML = parts.join('<span class="sep"> | </span>');
  $("ctxChip").classList.toggle("live", shown.length > 0);
}

/* ------------------------------------------------------------ boot */
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
  syncM3Select($("dispMode"));
  syncM3Select($("dispCmap"));
  $("modelPath").value = data.defaults.unet_model || "";
  await reloadSequences();
}

async function reloadSequences(refresh = false) {
  const suffix = refresh ? "&refresh=1" : "";
  const seqs = await apiJson(`/api/sequences?runs_root=${encodeURIComponent(state.runsRoot)}${suffix}`);
  const sel = $("seqSelect");
  sel.innerHTML = "";
  for (const item of seqs.sequences) {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.textContent = item.label;
    opt.title = item.path;
    sel.appendChild(opt);
  }
  if (!seqs.sequences.length) {
    syncM3Select(sel);
    $("stageEmpty").classList.remove("hidden");
    $("stageEmpty").textContent = "runs/ 下没有找到 frames.npy，请检查目录或在左侧粘贴路径。";
    return;
  }
  const preferred = seqs.sequences.find((s) => s.label.includes("结题演示·优先")) || seqs.sequences[0];
  sel.value = preferred.path;
  syncM3Select(sel);
  await loadSequence(preferred.path);
}

/* ------------------------------------------------------------ input */
async function loadSequence(path) {
  if (!path) return;
  state.kind = "sequence";
  state.seq = path;
  state.index = 0;
  state.quadCache.clear();
  state.strip = null;
  state.stripKey = "";
  state.videoResult = null;
  stopPlayback();

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
          label: "已有结果",
          source: match ? match.label : "",
          detect: auto.predictions,
          track: auto.tracks || null,
          evaluate: auto.metrics || null,
        }]
      : [];
    state.slots = padSlots(state.results.length ? [0] : []);
    state.quadSlot = 0;
    await refreshMeta();
    await render();
  } catch (err) {
    $("stageEmpty").classList.remove("hidden");
    $("stageEmpty").textContent = `载入失败：${err.message}`;
  }
}

async function refreshMeta() {
  const first = state.results[state.slots.find((i) => i >= 0)] || state.results[0];
  try {
    state.meta = await apiJson("/api/browser/meta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sequence_dir: state.seq,
        detect_dir: first?.detect || null,
        track_dir: first?.track || null,
        evaluate_dir: first?.evaluate || null,
      }),
    });
    state.count = state.meta.count || 0;
  } catch {
    state.meta = null;
    state.count = 0;
  }
  updateCtx();
}

/* ------------------------------------------------------------ render */
async function render() {
  closeM3Menu(false);
  if (state.kind === "video") {
    renderVideo();
    return;
  }
  if (!state.results.length) {
    ["columns", "quad", "quadBar", "pills", "strip", "videoStage", "colHint"].forEach((id) =>
      $(id).classList.add("hidden")
    );
    $("stageEmpty").classList.remove("hidden");
    $("stageEmpty").textContent = "该序列还没有检测结果。在左侧勾选模型并点「运行检测」。";
    updateCtx();
    return;
  }
  $("stageEmpty").classList.add("hidden");
  $("videoStage").classList.add("hidden");

  if (state.view === "quad") {
    $("columns").classList.add("hidden");
    $("pills").classList.add("hidden");
    $("colHint").classList.add("hidden");
    $("quad").classList.remove("hidden");
    $("quadBar").classList.remove("hidden");
    await renderQuad();
  } else {
    $("quad").classList.add("hidden");
    $("quadBar").classList.add("hidden");
    $("columns").classList.remove("hidden");
    $("pills").classList.remove("hidden");
    await renderColumns();
  }
  await renderStrip();
  updateCtx();
}

function frameLabel() {
  $("frameLabel").textContent = state.count
    ? `帧 ${state.index + 1} / ${state.count}`
    : "帧 0 / 0";
}

const SLOT_COUNT = 3;

function slotOptions(current) {
  const list = state.results
    .map((r, idx) => `<option value="${idx}"${current === idx ? " selected" : ""}>${esc(labelOf(r))}</option>`)
    .join("");
  const empty = current < 0 ? " selected" : "";
  return `<option value="-1"${empty}>（空位）</option>${list}`;
}

function openSlotMenu(colEl, slot) {
  const wasOpen = colEl.classList.contains("menu-open");
  closeSlotMenus();
  closeM3Menu(false);
  if (wasOpen) return;

  colEl.classList.add("menu-open");
  const menu = document.createElement("div");
  menu.className = "slot-menu";
  menu.setAttribute("role", "menu");

  const title = document.createElement("div");
  title.className = "slot-menu-title";
  title.textContent = "选择对比结果";
  menu.appendChild(title);

  state.results.forEach((r, idx) => {
    const usedIn = state.slots.indexOf(idx);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "slot-menu-item";
    btn.innerHTML = `<span>${esc(labelOf(r))}</span>${usedIn >= 0 ? `<span class="tag">槽位 ${usedIn + 1}</span>` : ""}`;
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      closeSlotMenus();
      selectSlotValue(slot, idx);
    });
    menu.appendChild(btn);
  });

  if (state.results.length <= 1) {
    const hint = document.createElement("div");
    hint.className = "slot-menu-hint";
    hint.textContent = "在左侧勾选更多模型并点击「运行检测」，即可展开多列并排对比。";
    menu.appendChild(hint);
  }

  colEl.appendChild(menu);
}

async function renderColumns() {
  const filled = state.slots
    .map((idx, slot) => ({ slot, idx }))
    .filter((x) => x.idx >= 0 && state.results[x.idx]);

  let data = null;
  if (filled.length) {
    try {
      data = await apiJson("/api/compare/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence_dir: state.seq,
          sides: filled.map((x) => {
            const r = state.results[x.idx];
            return { detect_dir: r.detect, evaluate_dir: r.evaluate || null, label: labelOf(r) };
          }),
          index: state.index,
          settings: getSettings(),
        }),
      });
    } catch (err) {
      $("stageEmpty").classList.remove("hidden");
      $("stageEmpty").textContent = err.message;
      return;
    }
    state.count = data.count;
    state.index = data.index;
    frameLabel();
  }

  const bySlot = new Map();
  if (data) filled.forEach((x, k) => bySlot.set(x.slot, data.sides[k]));
  state.sides = [...bySlot.values()];
  $("columns").classList.toggle("pixelated", $("dispPixelated").checked);

  const cells = [];
  for (let slot = 0; slot < SLOT_COUNT; slot++) {
    const side = bySlot.get(slot);
    const current = state.slots[slot] ?? -1;
    const head = `
      <div class="col-head">
        <select class="slotPick" data-slot="${slot}" aria-label="选择槽位 ${slot + 1} 对比结果">${slotOptions(current)}</select>
        <span class="f1">${side ? `F1 ${fmt(side.summary?.target_f1)}` : ""}</span>
      </div>`;
    if (!side) {
      cells.push(`
        <div class="col col-empty" data-slot="${slot}" title="点击添加对比结果">
          ${head}
          <button class="slot-empty" data-slot="${slot}" type="button">
            <span class="slot-plus" aria-hidden="true">+</span>
            <span class="slot-label">添加对比</span>
            <span class="slot-sub">选择已有结果，或在左侧勾选模型运行</span>
          </button>
          <div class="col-foot"></div>
        </div>`);
      continue;
    }
    const s = side.summary || {};
    cells.push(`
      <div class="col">
        ${head}
        <div class="col-img"><img src="${side.image}" alt="${esc(side.label)}" /></div>
        <div class="col-foot">
          <span>IoU <span class="hot">${fmt(s.iou)}</span></span>
          <span>虚警/帧 <span class="hot">${fmt(s.fp_per_frame, 2)}</span></span>
          <span>重叠 <span class="hot">${esc(side.overlap)}</span> px</span>
        </div>
      </div>`);
  }
  $("columns").innerHTML = cells.join("");

  for (const sel of document.querySelectorAll("#columns .slotPick")) {
    sel.addEventListener("change", (ev) => {
      ev.stopPropagation();
      selectSlotValue(Number(ev.target.dataset.slot), Number(ev.target.value));
    });
    sel.addEventListener("click", (ev) => ev.stopPropagation());
  }
  upgradeAllSelects($("columns"));

  for (const box of document.querySelectorAll("#columns .slot-empty")) {
    box.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const slot = Number(box.dataset.slot);
      const colEl = box.closest(".col-empty");
      if (colEl) openSlotMenu(colEl, slot);
    });
  }

  // 空槽位已收纳为紧凑的「+ 添加对比」侧条，仅在全部为空时显示底部提示以节省主舞台高度
  const hint = $("colHint");
  if (filled.length === 0) {
    hint.classList.remove("hidden");
    hint.textContent = "在左侧勾选模型后点击「运行检测」，或点击空槽位选择已有结果。";
  } else {
    hint.classList.add("hidden");
  }
  renderPills(state.sides);
  renderMetricTable(state.sides);
}

function pillCard(label, value, delta, digits = 3) {
  let tail = "";
  if (delta !== null && delta !== undefined) {
    if (Math.abs(delta) < 1e-9) {
      tail = `<em style="color:var(--on-3);">±0</em>`;
    } else {
      const cls = delta > 0 ? "up" : "down";
      const arrow = delta > 0 ? "↑" : "↓";
      tail = `<em class="${cls}">${arrow}${Math.abs(delta).toFixed(digits)}</em>`;
    }
  }
  return `<div class="pillcard"><span>${label}</span><b>${value}${tail}</b></div>`;
}

function renderPills(sides) {
  const first = sides[0]?.summary || {};
  const last = sides[sides.length - 1]?.summary || {};
  const diff = (key) => {
    if (sides.length < 2) return null;
    const a = first[key];
    const b = last[key];
    return Number.isFinite(a) && Number.isFinite(b) ? b - a : null;
  };
  const host = $("pills");
  if (!sides.length || !sides[0].summary) {
    host.classList.add("hidden");
    return;
  }
  host.classList.remove("hidden");
  host.classList.toggle("has-delta", sides.length >= 2);
  host.innerHTML = [
    pillCard("目标级 F1", fmt(last.target_f1), diff("target_f1"), 3),
    pillCard("区域 IoU", fmt(last.iou), diff("iou"), 3),
    pillCard("每帧虚警", fmt(last.fp_per_frame, 2), diff("fp_per_frame"), 2),
    pillCard("匹配帧", fmt(last.matched, 0), diff("matched"), 0),
  ].join("");
}

function renderMetricTable(sides) {
  if (!sides.some((s) => s.summary)) {
    $("metricTable").innerHTML = `<tr><td style="color:var(--on-3);">暂无指标</td></tr>`;
    return;
  }
  const base = sides[0]?.summary || {};
  const head = `<thead><tr><th>指标</th>${sides
    .map((s) => `<th>${esc(s.label)}</th>`)
    .join("")}</tr></thead>`;
  const body = METRIC_ROWS.map((row) => {
    const cells = sides.map((side, i) => {
      const v = side.summary ? side.summary[row.key] : null;
      let delta = "";
      if (i > 0 && side.summary) {
        const b = base[row.key];
        if (Number.isFinite(b) && Number.isFinite(v)) {
          const d = v - b;
          if (Math.abs(d) > 1e-9) {
            const good = row.better === "down" ? d < 0 : d > 0;
            delta = `<span class="delta ${good ? "up" : "down"}">${good ? "↑" : "↓"}${Math.abs(d).toFixed(row.digits)}</span>`;
          } else {
            delta = `<span class="delta flat">±0</span>`;
          }
        }
      }
      return `<td class="val">${fmt(v, row.digits)}${delta}</td>`;
    });
    return `<tr><td>${row.label}</td>${cells.join("")}</tr>`;
  }).join("");
  $("metricTable").innerHTML = `${head}<tbody>${body}</tbody>`;
}

async function renderQuad() {
  const idx = state.slots[state.quadSlot] >= 0 ? state.slots[state.quadSlot] : state.slots.find((i) => i >= 0);
  const result = state.results[idx];
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
      $("stageEmpty").classList.remove("hidden");
      $("stageEmpty").textContent = err.message;
      return;
    }
    state.quadCache.set(key, data);
    if (state.quadCache.size > 60) state.quadCache.clear();
  }
  state.count = data.count;
  state.index = data.index;
  frameLabel();

  const panels = [
    ["1 · 增强 SAR 原图", "ENHANCED"],
    ["2 · 时序差分", data.index === 0 ? "首帧无历史" : "TEMP-DIFF"],
    ["3 · 检测与轨迹", "TRACKING"],
    ["4 · 质心 4× 放大", "ROI ×4"],
  ];
  $("quad").innerHTML = data.panels
    .map((src, i) => {
      // The difference view has no reference frame yet; a black tile reads as a bug.
      const blank = i === 1 && data.index === 0;
      const body = blank
        ? `<div class="tile-empty">首帧无历史参考帧<br />切到第 2 帧后可见</div>`
        : `<img src="${src}" alt="" />`;
      return `
      <figure>
        <figcaption><span>${panels[i][0]}</span><span>${panels[i][1]}</span></figcaption>
        <div class="col-img">${body}</div>
      </figure>`;
    })
    .join("");
  $("quad").classList.toggle("pixelated", $("dispPixelated").checked);

  const sel = $("quadSource");
  sel.innerHTML = state.slots
    .map((slotIdx, i) => `<option value="${i}"${i === state.quadSlot ? " selected" : ""}>${esc(labelOf(state.results[slotIdx]))}</option>`)
    .join("");
  sel.classList.toggle("hidden", state.slots.length < 2);
  syncM3Select(sel);
  if (state.sides.length) renderPills(state.sides);
}

async function renderStrip() {
  if (state.kind === "video" || !state.results.length) {
    $("strip").classList.add("hidden");
    return;
  }
  const result = state.results[state.slots.find((i) => i >= 0)];
  const key = `${state.seq}|${result?.detect || ""}|${settingsKey()}`;
  if (state.stripKey !== key || !state.strip) {
    try {
      state.strip = await apiJson("/api/browser/strip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sequence_dir: state.seq,
          detect_dir: result?.detect || null,
          settings: getSettings(),
          width: 96,
        }),
      });
      state.stripKey = key;
    } catch {
      $("strip").classList.add("hidden");
      return;
    }
  }
  const host = $("strip");
  host.classList.remove("hidden");
  host.innerHTML = state.strip.thumbs
    .map((src, i) => `<button data-frame="${i}" class="${i === state.index ? "on" : ""}" title="第 ${i + 1} 帧">
        <img src="${src}" alt="" /><i>${i + 1}</i></button>`)
    .join("");
  for (const btn of host.querySelectorAll("button")) {
    btn.addEventListener("click", () => gotoFrame(Number(btn.dataset.frame)));
  }
  markStrip();
}

function markStrip() {
  const host = $("strip");
  if (!host || host.classList.contains("hidden")) return;
  host.querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === state.index));
  const active = host.querySelector("button.on");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function closeSheet() {
  if ($("metricSheet").classList.contains("hidden")) return;
  $("metricSheet").classList.add("hidden");
  $("btnAllMetrics").focus();   // 焦点回入口，键盘用户不会丢失位置
}

function renderVideo() {
  ["columns", "quad", "quadBar", "pills", "strip", "stageEmpty", "colHint"].forEach((id) =>
    $(id).classList.add("hidden")
  );
  $("videoStage").classList.remove("hidden");
  const res = state.videoResult;
  if (!res) {
    $("videoOverlay").classList.add("hidden");
    $("videoDownloads").innerHTML = `<span class="lbl">上传视频并点「运行检测」后，这里显示叠加结果。</span>`;
    return;
  }
  $("videoOverlay").classList.remove("hidden");
  if (res.overlay_url) $("videoOverlay").src = `${res.overlay_url}&t=${Date.now()}`;
  const links = [];
  if (res.overlay_download) links.push(`<a class="btn btn-sm" href="${res.overlay_download}">下载叠加视频</a>`);
  if (res.csv_download) links.push(`<a class="btn btn-sm" href="${res.csv_download}">下载 predictions.csv</a>`);
  $("videoDownloads").innerHTML = links.join("");
}

/* ------------------------------------------------------------ playback */
function stopPlayback() {
  state.playing = false;
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }
  setPlayState(false);
}

function togglePlayback() {
  if (state.playing) { stopPlayback(); return; }
  if (state.count <= 1) return;
  state.playing = true;
  setPlayState(true);
  const step = async () => {
    if (!state.playing) return;
    const t0 = performance.now();
    state.index = (state.index + 1) % state.count;
    try {
      await (state.view === "quad" ? renderQuad() : renderColumns());
      markStrip();
    } catch { stopPlayback(); return; }
    const interval = 1000 / Number($("fps").value || 8);
    state.timer = setTimeout(step, Math.max(0, interval - (performance.now() - t0)));
  };
  step();
}

async function gotoFrame(index) {
  stopPlayback();
  if (!state.count) return;
  state.index = Math.max(0, Math.min(state.count - 1, index));
  await (state.view === "quad" ? renderQuad() : renderColumns());
  markStrip();
}

/* ------------------------------------------------------------ run */
const METHOD_NAMES = { threshold: "阈值", difference: "差分", learned: "U-Net" };
const VERB_NAMES = { detect: "检测", track: "跟踪", evaluate: "评估", "detect-video": "逐帧推理" };

function resetSteps() {
  $("logWrap").classList.remove("hidden", "collapsed");
  $("log").classList.add("hidden");
  $("log").textContent = "";
  $("steps").innerHTML = "";
  $("runDots").classList.remove("done");
  $("runTitle").textContent = "正在运行…";
  state.stepNo = 0;
}

function addStep(text, cls = "") {
  const host = $("steps");
  const prev = host.lastElementChild;
  // 新一步出现时，上一步转为"已完成"的灰字，跟录屏里的节奏一致
  if (prev && !prev.classList.contains("group")) prev.classList.add("done");
  const el = document.createElement("div");
  el.className = `step ${cls}`.trim();
  state.stepNo += 1;
  const idx = document.createElement("span");
  idx.className = "idx";
  idx.textContent = String(state.stepNo).padStart(2, "0");
  const body = document.createElement("span");
  body.className = "txt";
  body.textContent = text;
  el.append(idx, body);
  host.appendChild(el);
  const scroller = $("steps").parentElement;
  if (scroller) scroller.scrollTop = scroller.scrollHeight;
  return el;
}

function stepLine(raw) {
  const line = String(raw).trim();
  if (!line) return;
  if (line.startsWith("===")) {
    addStep(line.replace(/=/g, "").trim(), "group");
    return;
  }
  if (line.startsWith("> videosar ")) {
    const verb = line.split(/\s+/)[2] || "";
    const method = (line.match(/--method (\w+)/) || [])[1];
    const label = VERB_NAMES[verb] || verb;
    addStep(method ? `${label} · ${METHOD_NAMES[method] || method}` : label);
    return;
  }
  if (/^IoU:/.test(line)) { addStep(line); return; }
  if (/^\[退出码/.test(line)) {
    const last = $("steps").lastElementChild;
    if (last) last.classList.add("done");
    return;
  }
  if (/失败|Error|Traceback|异常/.test(line)) addStep(line);
}

function streamJob(jobId, onDone) {
  resetSteps();
  setBadge($("runBadge"), "运行中", "warn");
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.addEventListener("log", (ev) => {
    const { line } = JSON.parse(ev.data);
    $("log").textContent += (($("log").textContent) ? "\n" : "") + line;
    stepLine(line);
  });
  es.addEventListener("done", (ev) => {
    es.close();
    const { ok, result } = JSON.parse(ev.data);
    setBadge($("runBadge"), ok ? "已完成" : "执行失败", ok ? "ok" : "err");
    $("runDots").classList.add("done");
    $("runTitle").textContent = ok ? "已完成" : "执行失败";
    $("steps").querySelectorAll(".step").forEach((el) => el.classList.add("done"));
    // 跑完收起成一行，把画布高度还给结果
    setTimeout(() => $("logWrap").classList.add("collapsed"), 1400);
    onDone(ok, result);
  });
  es.onerror = () => {
    es.close();
    setBadge($("runBadge"), "连接中断", "err");
    $("runTitle").textContent = "连接中断";
  };
}

async function runDetection() {
  if (state.kind === "video") return runVideoDetection();
  const methods = [...state.methods];
  if (!methods.length) { alert("请至少选择一个模型"); return; }
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
      state.slots = padSlots(state.results.slice(0, SLOT_COUNT).map((_, i) => i));
      state.quadSlot = 0;
      state.index = 0;
      state.quadCache.clear();
      state.stripKey = "";
      await refreshMeta();
      await render();
    });
  } catch (err) {
    $("btnRun").disabled = false;
    setBadge($("runBadge"), "启动失败", "err");
    $("logWrap").classList.remove("hidden");
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
      renderVideo();
    });
  } catch (err) {
    $("btnRun").disabled = false;
    setBadge($("runBadge"), "启动失败", "err");
    $("logWrap").classList.remove("hidden");
    $("log").classList.remove("hidden");
    $("log").textContent = err.message;
  }
}

/* ------------------------------------------------------------ video crop */
function setKind(kind) {
  state.kind = kind;
  [...$("kindSeg").children].forEach((b) => b.classList.toggle("on", b.dataset.kind === kind));
  $("seqPane").classList.toggle("hidden", kind !== "sequence");
  $("videoPane").classList.toggle("hidden", kind !== "video");
  $("cropSection").classList.toggle("hidden", kind !== "video" || !state.videoInfo);
  if (kind === "video") {
    state.results = [];
    state.slots = padSlots([]);
    state.count = 0;
    state.videoResult = null;
    stopPlayback();
    frameLabel();
    renderVideo();
  } else if (state.seq) {
    loadSequence(state.seq);
  }
  updateCtx();
}

function loadCropFrame(info) {
  state.video = info.video_path;
  state.videoInfo = info;
  state.crop = { x: 0, y: 0, w: info.width, h: info.height };
  $("dimLabel").textContent = `${info.width} × ${info.height}`;
  $("cropSection").classList.remove("hidden");
  syncCrop();
  const img = new Image();
  img.onload = () => { state.cropImg = img; drawCrop(); };
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
  ctx.fillStyle = "rgba(0,0,0,.38)";
  ctx.fillRect(0, 0, canvas.width, y);
  ctx.fillRect(0, y + h, canvas.width, canvas.height - (y + h));
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, canvas.width - (x + w), h);
  ctx.strokeStyle = "#ff4040";
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 260));
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

/* ------------------------------------------------------------ events */
function bind() {
  upgradeAllSelects(document);

  $("kindSeg").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-kind]");
    if (btn) setKind(btn.dataset.kind);
  });

  $("seqSelect").addEventListener("change", (e) => loadSequence(e.target.value));
  $("seqPath").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadSequence(e.target.value.trim());
  });
  $("btnReloadRuns").addEventListener("click", () => reloadSequences(true));

  const pick = async (kind, btn) => {
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "打开窗口中…";
    try {
      const { path } = await apiJson(`/api/dialog/${kind}`, { method: "POST" });
      if (path) {
        // A picked frames.npy lives inside the sequence folder.
        const target = path.toLowerCase().endsWith(".npy") ? path.replace(/[\\/][^\\/]+$/, "") : path;
        $("seqPath").value = target;
        const opt = [...$("seqSelect").options].find((o) => o.value === target);
        if (opt) {
          $("seqSelect").value = target;
          syncM3Select($("seqSelect"));
        }
        await loadSequence(target);
      }
    } catch (err) {
      setBadge($("runBadge"), err.message, "err");
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  };
  $("btnPickFolder").addEventListener("click", (ev) => pick("folder", ev.currentTarget));
  $("btnPickFile").addEventListener("click", (ev) => pick("file", ev.currentTarget));

  $("videoFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const nameEl = $("videoFileName");
    const boxEl = $("videoFileBox");
    if (nameEl) nameEl.textContent = file.name;
    if (boxEl) boxEl.classList.add("has-file");
    const fd = new FormData();
    fd.append("file", file);
    setBadge($("runBadge"), "解码首帧", "warn");
    try {
      const info = await apiJson("/api/video/upload", { method: "POST", body: fd });
      loadCropFrame(info);
      setBadge($("runBadge"), "待运行");
    } catch (err) {
      setBadge($("runBadge"), err.message, "err");
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
    } catch (err) {
      setBadge($("runBadge"), err.message, "err");
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
    state.crop = {
      x: Math.min(state.dragStart.x, pt.x),
      y: Math.min(state.dragStart.y, pt.y),
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

  $("methodList").addEventListener("click", (ev) => {
    const row = ev.target.closest("button[data-method]");
    if (!row) return;
    const method = row.dataset.method;
    if (state.methods.has(method)) {
      if (state.methods.size === 1) return;
      state.methods.delete(method);
      row.classList.remove("on");
    } else {
      state.methods.add(method);
      row.classList.add("on");
    }
    $("modelField").classList.toggle("hidden", !state.methods.has("learned"));
  });

  $("btnRun").addEventListener("click", runDetection);
  $("btnRawLog").addEventListener("click", (ev) => {
    ev.stopPropagation();
    $("logWrap").classList.remove("collapsed");
    $("log").classList.toggle("hidden");
  });
  $("logWrap").querySelector(".run-head").addEventListener("click", () => {
    $("logWrap").classList.toggle("collapsed");
  });

  $("viewSeg").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-view]");
    if (!btn) return;
    [...$("viewSeg").children].forEach((b) => b.classList.toggle("on", b === btn));
    state.view = btn.dataset.view;
    stopPlayback();
    render();
  });

  $("btnPlay").addEventListener("click", togglePlayback);
  $("btnPrev").addEventListener("click", () => gotoFrame(state.index - 1));
  $("btnNext").addEventListener("click", () => gotoFrame(state.index + 1));

  $("quadSource").addEventListener("change", (e) => {
    state.quadSlot = Number(e.target.value) || 0;
    state.quadCache.clear();
    renderQuad();
  });

  $("btnDisplay").addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeM3Menu(false);
    $("displayPop").classList.toggle("hidden");
  });
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest(".pop-wrap") && !ev.target.closest(".m3-menu")) $("displayPop").classList.add("hidden");
    if (!ev.target.closest(".col-empty")) closeSlotMenus();
    if (!ev.target.closest(".m3-select") && !ev.target.closest(".m3-menu")) closeM3Menu(false);
  });
  window.addEventListener("resize", () => closeM3Menu(false));
  document.querySelector(".rail-body")?.addEventListener("scroll", () => closeM3Menu(false), { passive: true });

  ["dispMode", "dispCmap", "dispDiff", "dispDenoise", "dispTruth", "dispPred"].forEach((id) => {
    $(id).addEventListener("change", () => {
      state.quadCache.clear();
      state.stripKey = "";
      state.strip = null;
      if (state.results.length) render();
    });
  });
  $("dispPixelated").addEventListener("change", () => {
    const on = $("dispPixelated").checked;
    $("columns").classList.toggle("pixelated", on);
    $("quad").classList.toggle("pixelated", on);
  });

  $("btnAllMetrics").addEventListener("click", () => {
    const video = $("videoOverlay");
    if (video && !video.paused) video.pause();
    $("metricSheet").classList.remove("hidden");
    $("btnCloseSheet").focus({ preventScroll: true });
  });
  $("btnCloseSheet").addEventListener("click", closeSheet);
  $("metricSheet").addEventListener("click", (ev) => {
    if (ev.target === $("metricSheet")) closeSheet();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (activeM3Menu) {
        e.preventDefault();
        closeM3Menu(true);
        return;
      }
      closeSlotMenus();
      if (!$("metricSheet").classList.contains("hidden")) {
        e.preventDefault();
        closeSheet();
        return;
      }
    }
    // 弹层是模态的：开着时不允许空格/方向键改背后的帧
    if (!$("metricSheet").classList.contains("hidden")) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    // 焦点在按钮上时，空格是原生激活，不能被劫持成播放
    if (e.code === "Space" && document.activeElement?.tagName === "BUTTON") return;
    if (!state.results.length) return;
    if (e.code === "Space") {
      if (state.count <= 1) return;   // 单帧时 togglePlayback 会直接返回，别白吞按键
      e.preventDefault();
      togglePlayback();
    }
    else if (e.code === "ArrowLeft") { e.preventDefault(); gotoFrame(state.index - 1); }
    else if (e.code === "ArrowRight") { e.preventDefault(); gotoFrame(state.index + 1); }
  });

  const exportQuad = async (format) => {
    const result = state.results[state.slots[state.quadSlot] >= 0 ? state.slots[state.quadSlot] : state.slots.find((i) => i >= 0)];
    if (!result) return;
    $("exportStatus").textContent = `正在渲染 ${format.toUpperCase()}…`;
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
      $("exportStatus").innerHTML = `<a class="btn btn-sm" href="${res.download_url}">下载 ${esc(res.filename)}</a>`;
    } catch (err) {
      $("exportStatus").textContent = err.message;
    }
  };
  $("btnGif").addEventListener("click", () => exportQuad("gif"));
  $("btnMp4").addEventListener("click", () => exportQuad("mp4"));
}

bind();
boot().catch((err) => {
  const el = $("stageEmpty");
  if (el) { el.classList.remove("hidden"); el.textContent = `初始化失败：${err.message}`; }
});