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
  v2: { runs: [], runId: "", data: null },
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

  const obs = new MutationObserver(() => syncM3Select(sel));
  obs.observe(sel, { childList: true, attributes: true, attributeFilter: ["class", "disabled"] });

  syncM3Select(sel);
}

function upgradeAllSelects(root = document) {
  root.querySelectorAll("select").forEach((sel) => {
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
  v2ModeChanged();
  // v2 历史列表独立容错：即使 /api/v2/runs 暂不可用也不覆盖主工作台
  try {
    await loadV2Runs();
  } catch (err) {
    console.warn("loadV2Runs skipped:", err);
  }
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
  setV2Mode(false);
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
    const empty = $("splitStage");
    if (empty) empty.classList.add("hidden");
    $("stageEmpty").classList.remove("hidden");
    $("stageEmpty").textContent = "该序列还没有检测结果。在左侧勾选模型并点「运行检测」。";
    updateCtx();
    return;
  }
  $("stageEmpty").classList.add("hidden");
  $("videoStage").classList.add("hidden");
  const splitHost = $("splitStage");
  if (splitHost) splitHost.classList.add("hidden");

  if (state.view === "quad") {
    $("columns").classList.add("hidden");
    $("pills").classList.add("hidden");
    $("colHint").classList.add("hidden");
    $("quad").classList.remove("hidden");
    $("quadBar").classList.remove("hidden");
    await renderQuad();
  } else if (state.view === "split") {
    $("columns").classList.add("hidden");
    $("pills").classList.add("hidden");
    $("colHint").classList.add("hidden");
    $("quad").classList.add("hidden");
    $("quadBar").classList.add("hidden");
    await renderSplit();
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

/* T02 帧缓存：LRU 40 帧，键 = 序列 | 槽位组合 | 帧号 | 显示设置 */
const COL_CACHE_MAX = 40;
const colCache = new Map();

function cacheGet(key) {
  if (!colCache.has(key)) return null;
  const value = colCache.get(key);
  colCache.delete(key);
  colCache.set(key, value);          // 命中即刷新热度
  return value;
}

function cacheSet(key, value) {
  if (colCache.has(key)) colCache.delete(key);
  colCache.set(key, value);
  while (colCache.size > COL_CACHE_MAX) colCache.delete(colCache.keys().next().value);
}

function columnSig(filled) {
  return [state.seq, filled.map((x) => `${x.slot}:${x.idx}`).join(","), settingsKey()].join("|");
}

function sidePayload(filled, index) {
  return {
    sequence_dir: state.seq,
    sides: filled.map((x) => {
      const r = state.results[x.idx];
      return { detect_dir: r.detect, evaluate_dir: r.evaluate || null, label: labelOf(r) };
    }),
    index,
    settings: getSettings(),
  };
}

async function fetchFrame(filled, index) {
  const key = `${columnSig(filled)}|${index}`;
  const hit = cacheGet(key);
  if (hit) return hit;
  const data = await apiJson("/api/compare/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sidePayload(filled, index)),
  });
  cacheSet(key, data);
  return data;
}

/* 空闲时预热相邻帧，播放与步进才不会每次都等网络往返 */
function prefetchNeighbors(filled) {
  if (!filled.length || state.count <= 1) return;
  const targets = [state.index - 1, state.index + 1].filter((i) => i >= 0 && i < state.count);
  if (!targets.length) return;
  const run = () => targets.forEach((i) => fetchFrame(filled, i).catch(() => {}));
  if (typeof window.requestIdleCallback === "function") window.requestIdleCallback(run, { timeout: 900 });
  else setTimeout(run, 160);
}

const ICON_EXPAND = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 3H5a2 2 0 0 0-2 2v4M15 3h4a2 2 0 0 1 2 2v4M9 21H5a2 2 0 0 1-2-2v-4M15 21h4a2 2 0 0 0 2-2v-4"/></svg>';

function colHeadHtml(slot, current, side, sideNo) {
  return `
      <div class="col-head">
        <select class="slotPick" data-slot="${slot}" aria-label="选择槽位 ${slot + 1} 对比结果">${slotOptions(current)}</select>
        <span class="f1">${side ? `F1 ${fmt(side.summary?.target_f1)}` : ""}</span>
        ${side ? `<button class="col-zoom-btn" data-zoom-side="${sideNo}" type="button" title="放大 (F)" aria-label="放大第 ${slot + 1} 栏">${ICON_EXPAND}</button>` : ""}
      </div>`;
}

function colFootHtml(side) {
  const s = side.summary || {};
  return `
        <div class="col-foot">
          <span>IoU <span class="hot" data-metric="iou">${fmt(s.iou)}</span></span>
          <span>虚警/帧 <span class="hot" data-metric="fp_per_frame">${fmt(s.fp_per_frame, 2)}</span></span>
          <span>重叠 <span class="hot">${esc(side.overlap)}</span> px</span>
        </div>`;
}

function rebuildColumns(bySlot) {
  const cells = [];
  let sideNo = -1;
  for (let slot = 0; slot < SLOT_COUNT; slot++) {
    const side = bySlot.get(slot);
    const current = state.slots[slot] ?? -1;
    if (!side) {
      cells.push(`
        <div class="col col-empty" data-slot="${slot}" title="点击添加对比结果">
          ${colHeadHtml(slot, current, null, -1)}
          <button class="slot-empty" data-slot="${slot}" type="button">
            <span class="slot-plus" aria-hidden="true">+</span>
            <span class="slot-label">添加对比</span>
            <span class="slot-sub">选择已有结果，或在左侧勾选模型运行</span>
          </button>
          <div class="col-foot"></div>
        </div>`);
      continue;
    }
    sideNo += 1;
    cells.push(`
      <div class="col" data-slot="${slot}" data-side="${sideNo}">
        ${colHeadHtml(slot, current, side, sideNo)}
        <div class="col-img">
          <img src="${side.image}" alt="${esc(side.label)}" draggable="false" />
          <span class="xhair-h" aria-hidden="true"></span>
          <span class="xhair-v" aria-hidden="true"></span>
          <span class="zoom-badge" aria-hidden="true">1.0×</span>
        </div>
        ${colFootHtml(side)}
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
}

/* 只换 src 与指标文字，不重建 DOM：图片不闪、缩放与十字准星也不丢 */
async function patchColumns(bySlot, filled) {
  const cols = [...document.querySelectorAll("#columns .col")];
  if (cols.length !== SLOT_COUNT) return false;
  if (cols.filter((c) => c.dataset.side !== undefined).length !== filled.length) return false;

  const jobs = [];
  for (let slot = 0; slot < SLOT_COUNT; slot++) {
    const side = bySlot.get(slot);
    if (!side) continue;
    const col = cols.find((c) => Number(c.dataset.slot) === slot && c.dataset.side !== undefined);
    if (!col) return false;
    const box = col.querySelector(".col-img");
    const img = box?.querySelector("img");
    if (!box || !img) return false;

    if (img.getAttribute("src") !== side.image) {
      // 骨架屏只在真的等图时出现，缓存命中时不会闪一下
      const timer = setTimeout(() => box.classList.add("skeleton"), 130);
      jobs.push(
        (async () => {
          img.src = side.image;
          img.alt = side.label;
          try { await img.decode(); } catch { /* 解码失败也照常收尾 */ }
          clearTimeout(timer);
          box.classList.remove("skeleton");
        })()
      );
    }
    const s = side.summary || {};
    const f1 = col.querySelector(".col-head .f1");
    if (f1) f1.textContent = `F1 ${fmt(s.target_f1)}`;
    col.querySelectorAll(".col-foot .hot[data-metric]").forEach((el) => {
      const digits = el.dataset.metric === "fp_per_frame" ? 2 : 3;
      el.textContent = fmt(s[el.dataset.metric], digits);
    });
  }
  await Promise.all(jobs);
  return true;
}

async function renderColumns() {
  const filled = state.slots
    .map((idx, slot) => ({ slot, idx }))
    .filter((x) => x.idx >= 0 && state.results[x.idx]);

  let data = null;
  if (filled.length) {
    try {
      data = await fetchFrame(filled, state.index);
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

  const sig = columnSig(filled);
  const patched = sig === ui.colSig && (await patchColumns(bySlot, filled));
  if (!patched) {
    rebuildColumns(bySlot);
    ui.colSig = sig;
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
  markBestColumns(state.sides);
  applyZoom();
  prefetchNeighbors(filled);
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

const BEST_KEYS = { target_f1: "up", iou: "up", fp_per_frame: "down" };

/* 并列第一全部高亮；全员打平则视为无最优，避免整行变绿 */
function bestIndices(sides, key, better) {
  if (sides.length < 2) return new Set();
  const vals = sides.map((s) => (s && s.summary ? s.summary[key] : null));
  const nums = vals.filter((v) => Number.isFinite(v));
  if (nums.length < 2) return new Set();
  const target = better === "down" ? Math.min(...nums) : Math.max(...nums);
  const out = new Set();
  vals.forEach((v, i) => { if (Number.isFinite(v) && Math.abs(v - target) < 1e-9) out.add(i); });
  return out.size >= nums.length ? new Set() : out;
}

function bestMap(sides) {
  const map = new Map();
  for (const [key, better] of Object.entries(BEST_KEYS)) map.set(key, bestIndices(sides, key, better));
  return map;
}

function markBestColumns(sides) {
  document.querySelectorAll("#columns .col[data-side]").forEach((col) => {
    col.querySelectorAll(".is-best").forEach((el) => el.classList.remove("is-best"));
  });
  if (sides.length < 2) return;
  const map = bestMap(sides);
  for (const [key, set] of map) {
    if (!set.size) continue;
    for (const i of set) {
      const col = document.querySelector(`#columns .col[data-side="${i}"]`);
      if (!col) continue;
      const target = key === "target_f1"
        ? col.querySelector(".col-head .f1")
        : col.querySelector(`.col-foot .hot[data-metric="${key}"]`);
      if (target) target.classList.add("is-best");
    }
  }
}

function renderMetricTable(sides) {
  if (!sides.some((s) => s.summary)) {
    $("metricTable").innerHTML = `<tr><td style="color:var(--on-3);">暂无指标</td></tr>`;
    return;
  }
  const base = sides[0]?.summary || {};
  const best = bestMap(sides);
  const head = `<thead><tr><th>指标</th>${sides
    .map((s) => `<th>${esc(s.label)}</th>`)
    .join("")}</tr></thead>`;
  const body = METRIC_ROWS.map((row) => {
    const winners = best.get(row.key);
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
      const hot = winners && winners.has(i) ? " is-best" : "";
      return `<td class="val${hot}">${fmt(v, row.digits)}${delta}</td>`;
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
  const statuses = Array.isArray(state.strip.frame_status) ? state.strip.frame_status : [];
  const STATUS_TEXT = { ok: "命中", missed: "漏检", fp_high: "虚警偏高", unknown: "" };
  host.innerHTML = state.strip.thumbs
    .map((src, i) => {
      const st = statuses[i] || "";
      const note = STATUS_TEXT[st] ? ` · ${STATUS_TEXT[st]}` : "";
      const heat = st && st !== "unknown" ? `<span class="heat ${st}" aria-hidden="true"></span>` : "";
      return `<button data-frame="${i}" class="${i === state.index ? "on" : ""}" title="第 ${i + 1} 帧${note}">
        <img src="${src}" alt="" /><i>${i + 1}</i>${heat}</button>`;
    })
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
  const splitHost = $("splitStage");
  if (splitHost) splitHost.classList.add("hidden");
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

function paintFrame() {
  if (state.view === "quad") return renderQuad();
  if (state.view === "split") return renderSplit();
  return renderColumns();
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
      await paintFrame();
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
  await paintFrame();
  markStrip();
}

/* ------------------------------------------------------------ run */
const METHOD_NAMES = { threshold: "阈值", difference: "差分", learned: "U-Net" };
const VERB_NAMES = { detect: "检测", track: "跟踪", evaluate: "评估", "detect-video": "逐帧推理" };

function resetSteps() {
  $("logWrap").classList.remove("hidden", "collapsed");
  $("log").classList.add("hidden");
  $("log").textContent = "";
  $("btnRawLog").textContent = "原始日志";
  $("steps").innerHTML = "";
  $("runDots").classList.remove("done", "err");
  $("runTitle").textContent = "正在运行…";
  state.stepNo = 0;
}

function addStep(text, cls = "") {
  const host = $("steps");
  const prev = host.lastElementChild;
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
    $("runDots").classList.add(ok ? "done" : "err");
    const countText = state.stepNo ? ` · ${state.stepNo} 步` : "";
    $("runTitle").textContent = ok ? `已完成${countText}` : "执行失败";
    $("steps").querySelectorAll(".step").forEach((el) => el.classList.add("done"));
    // 跑完自动收成右下角小胶囊，失败时保留展开方便查看原因
    if (ok) setTimeout(() => $("logWrap").classList.add("collapsed"), 1400);
    onDone(ok, result);
  });
  es.onerror = () => {
    es.close();
    setBadge($("runBadge"), "连接中断", "err");
    $("runDots").classList.add("err");
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
    $("logWrap").classList.remove("hidden", "collapsed");
    $("runDots").classList.add("err");
    $("runTitle").textContent = "启动失败";
    $("log").classList.remove("hidden");
    $("btnRawLog").textContent = "隐藏日志";
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
    $("logWrap").classList.remove("hidden", "collapsed");
    $("runDots").classList.add("err");
    $("runTitle").textContent = "启动失败";
    $("log").classList.remove("hidden");
    $("btnRawLog").textContent = "隐藏日志";
    $("log").textContent = err.message;
  }
}

/* ------------------------------------------------------------ v2.0 benchmark contract
   The two partitions are rendered into separate cards from separate keys;
   there is deliberately no combined / averaged score anywhere in this view. */
const V2_METRIC_ROWS = [
  { key: "map_50", label: "mAP@50", digits: 3 },
  { key: "map_30", label: "mAP@30", digits: 3 },
  { key: "recall_at_1_fa", label: "Recall@1FA", digits: 3 },
  { key: "fa_per_frame_50", label: "FA/frame", digits: 3 },
  { key: "mota", label: "MOTA", digits: 3 },
  { key: "idf1", label: "IDF1", digits: 3 },
  { key: "hota", label: "HOTA", digits: 3 },
  { key: "id_switches", label: "ID Switches", digits: 0 },
];
const V2_FROC_COLORS = ["#0b57d0", "#b3261e", "#146c2e", "#8b5cf6", "#9a5b00", "#00838f", "#c2185b", "#5f6368"];

function isV2Mode() {
  return Boolean(document.querySelector(".stage")?.classList.contains("v2-mode"));
}

function setV2Mode(on) {
  const active = Boolean(on);
  if (active) stopPlayback();
  document.querySelector(".stage")?.classList.toggle("v2-mode", active);
  $("v2Panel")?.classList.toggle("hidden", !active);
  if (!active) $("v2FrocTip")?.classList.add("hidden");
}

function exitV2Mode() {
  state.v2.runId = "";
  state.v2.data = null;
  const sel = $("v2RunSelect");
  if (sel) {
    sel.value = "";
    syncM3Select(sel);
  }
  setV2Mode(false);
}

function v2ModeChanged() {
  const mode = $("v2Mode").value;
  const mwp = $("v2Mwp");
  mwp.disabled = mode !== "simulate";
  if (mwp.disabled) mwp.checked = false;
  $("v2DataRootField").classList.toggle("hidden", !mode.startsWith("eval-"));
}

async function loadV2Runs() {
  const data = await apiJson("/api/v2/runs");
  state.v2.runs = data.runs || [];
  const sel = $("v2RunSelect");
  const previous = sel.value;
  sel.innerHTML = '<option value="">— 选择历史运行 —</option>';
  for (const run of state.v2.runs) {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    const flags = [run.has_real ? "实测" : "", run.has_synthetic ? "仿真" : ""].filter(Boolean).join("+");
    opt.textContent = `${run.run_id} · ${run.mode}${flags ? " · " + flags : ""}`;
    opt.title = `${run.path}\n${run.timestamp}`;
    sel.appendChild(opt);
  }
  if (previous && [...sel.options].some((o) => o.value === previous)) sel.value = previous;
  syncM3Select(sel);
}

async function loadV2Run(runId) {
  state.v2.runId = runId || "";
  if (!runId) { state.v2.data = null; setV2Mode(false); return; }
  const encoded = String(runId).split("/").map(encodeURIComponent).join("/");
  const data = await apiJson(`/api/v2/runs/${encoded}`);
  state.v2.data = data;
  setV2Mode(true);
  renderV2(data);
}

function renderV2(data) {
  $("v2RunId").textContent = data.run_id || "";
  const meta = [];
  if (data.mode) meta.push(`模式 <span class="mono">${esc(data.mode)}</span>`);
  if (data.timestamp_utc) meta.push(`时间 <span class="mono">${esc(data.timestamp_utc)}</span>`);
  if (data.git_commit) meta.push(`git <span class="mono">${esc(String(data.git_commit).slice(0, 10))}</span>`);
  if (data.seed !== null && data.seed !== undefined) meta.push(`seed <span class="mono">${esc(data.seed)}</span>`);
  $("v2Meta").innerHTML = meta.join('<span class="sep"> | </span>');
  const summary = data.summary || [];
  renderV2Partition("v2RealBody", "v2RealCount", "real", summary, data);
  renderV2Partition("v2SynthBody", "v2SynthCount", "synthetic", summary, data);
  renderFroc(data.froc_curve || []);
}

function renderV2Partition(bodyId, countId, domain, summary, data) {
  const body = $(bodyId);
  const partition = domain === "real" ? (data.real_blind_eval || {}) : (data.synthetic_ablation || {});
  const names = new Set(Object.keys(partition));
  const rows = (summary || []).filter((r) => r.domain === domain || names.has(r.split_name));
  if (!rows.length) {
    $(countId).textContent = "0 split";
    body.innerHTML = '<div class="v2-empty">本次运行未包含该分区</div>';
    return;
  }
  $(countId).textContent = `${rows.length} split`;
  const head = ["split_name", "domain", ...V2_METRIC_ROWS.map((r) => r.label)];
  let html = '<table class="v2-metrics"><thead><tr>' + head.map((h) => `<th>${esc(h)}</th>`).join("") + '</tr></thead><tbody>';
  for (const row of rows) {
    html += `<tr><td>${esc(row.split_name)}</td><td>${esc(row.domain)}</td>`;
    for (const metric of V2_METRIC_ROWS) html += `<td>${fmt(row[metric.key], metric.digits)}</td>`;
    html += '</tr>';
  }
  body.innerHTML = html + '</tbody></table>';
}

function renderFroc(points) {
  const svg = $("v2Froc");
  const legend = $("v2FrocLegend");
  const tip = $("v2FrocTip");
  if (!svg || !legend) return;
  svg.innerHTML = "";
  legend.innerHTML = "";
  if (tip) tip.classList.add("hidden");
  if (!points.length) {
    svg.innerHTML = '<text x="360" y="150" text-anchor="middle" font-size="13" fill="#747775">本次运行无 FROC 曲线数据</text>';
    return;
  }
  const groups = new Map();
  points.forEach((point) => {
    const key = `${point.split_name}||${point.domain}`;
    if (!groups.has(key)) groups.set(key, { split: point.split_name, domain: point.domain, pts: [] });
    groups.get(key).pts.push(point);
  });
  const width = 720, height = 300, left = 64, right = 18, top = 16, bottom = 44;
  const pw = width - left - right, ph = height - top - bottom;
  const maxFa = Math.max(1, ...points.map((p) => Number(p.fa_per_frame) || 0));
  const px = (v) => left + (Math.max(0, Number(v) || 0) / maxFa) * pw;
  const py = (v) => top + (1 - Math.max(0, Math.min(1, Number(v) || 0))) * ph;
  const parts = [];
  for (let i = 0; i <= 4; i++) {
    const gy = top + (ph * i) / 4;
    parts.push(`<line x1="${left}" y1="${gy}" x2="${width - right}" y2="${gy}" stroke="#e1e5ec" stroke-width="1"/>`);
    parts.push(`<text x="${left - 8}" y="${gy + 4}" text-anchor="end" font-size="11" fill="#747775">${(1 - i / 4).toFixed(2)}</text>`);
  }
  for (let i = 0; i <= 5; i++) {
    const gx = left + (pw * i) / 5;
    parts.push(`<line x1="${gx}" y1="${top}" x2="${gx}" y2="${top + ph}" stroke="#e1e5ec" stroke-width="1"/>`);
    parts.push(`<text x="${gx}" y="${top + ph + 18}" text-anchor="middle" font-size="11" fill="#747775">${((maxFa * i) / 5).toFixed(2)}</text>`);
  }
  // 1.0 FA 参考虚线（呼应指标表中的 Recall@1FA）
  if (maxFa >= 1) {
    const x1 = px(1);
    parts.push(`<line x1="${x1.toFixed(2)}" y1="${top}" x2="${x1.toFixed(2)}" y2="${top + ph}" stroke="#0b57d0" stroke-opacity="0.35" stroke-width="1.2" stroke-dasharray="4 4"/>`);
    parts.push(`<text x="${Math.min(x1 + 4, width - right - 38).toFixed(2)}" y="${top + 12}" font-size="10.5" fill="#0b57d0" opacity="0.8">1.0 FA</text>`);
  }
  parts.push(`<line x1="${left}" y1="${top + ph}" x2="${width - right}" y2="${top + ph}" stroke="#c4c9d4" stroke-width="1"/>`);
  parts.push(`<line x1="${left}" y1="${top}" x2="${left}" y2="${top + ph}" stroke="#c4c9d4" stroke-width="1"/>`);
  parts.push(`<text x="${left + pw / 2}" y="${height - 6}" text-anchor="middle" font-size="11" fill="#444746">FA / frame</text>`);
  parts.push(`<text x="14" y="${top + ph / 2}" text-anchor="middle" font-size="11" fill="#444746" transform="rotate(-90 14 ${top + ph / 2})">Recall</text>`);
  let colorIndex = 0;
  for (const group of groups.values()) {
    const color = V2_FROC_COLORS[colorIndex % V2_FROC_COLORS.length];
    const ordered = [...group.pts].sort((a, b) => (Number(a.fa_per_frame) || 0) - (Number(b.fa_per_frame) || 0));
    const d = ordered.map((p, i) => `${i ? "L" : "M"}${px(p.fa_per_frame).toFixed(2)},${py(p.recall).toFixed(2)}`).join(" ");
    parts.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`);
    // 适度采样可悬停节点，避免点数过多时重叠拥挤
    const stride = Math.max(1, Math.ceil(ordered.length / 40));
    ordered.forEach((p, idx) => {
      if (idx % stride !== 0 && idx !== ordered.length - 1) return;
      const cx = px(p.fa_per_frame).toFixed(2);
      const cy = py(p.recall).toFixed(2);
      const faText = fmt(Number(p.fa_per_frame), 3);
      const rcText = fmt(Number(p.recall), 3);
      const label = esc(`${group.split} · ${group.domain}`);
      parts.push(`<circle class="froc-pt" cx="${cx}" cy="${cy}" r="3.2" fill="${color}" stroke="#ffffff" stroke-width="1.2" data-label="${label}" data-fa="${faText}" data-recall="${rcText}"><title>${label} | FA/frame ${faText} | Recall ${rcText}</title></circle>`);
    });
    legend.insertAdjacentHTML("beforeend", `<span class="lg"><span class="sw" style="background:${color}"></span>${esc(`${group.split} · ${group.domain}`)}</span>`);
    colorIndex++;
  }
  svg.innerHTML = parts.join("");

  if (tip) {
    const wrap = svg.parentElement;
    svg.querySelectorAll(".froc-pt").forEach((pt) => {
      pt.addEventListener("mouseenter", () => {
        const wRect = wrap.getBoundingClientRect();
        const pRect = pt.getBoundingClientRect();
        tip.innerHTML = `<b>${pt.dataset.label}</b><br/>FA/frame: ${pt.dataset.fa} · Recall: ${pt.dataset.recall}`;
        tip.style.left = `${Math.round(pRect.left - wRect.left + pRect.width / 2)}px`;
        tip.style.top = `${Math.round(pRect.top - wRect.top)}px`;
        tip.classList.remove("hidden");
      });
      pt.addEventListener("mouseleave", () => tip.classList.add("hidden"));
    });
  }
}

async function runV2Benchmark() {
  const mode = $("v2Mode").value;
  const body = { mode, mwp_enabled: mode === "simulate" && $("v2Mwp").checked };
  if (mode.startsWith("eval-")) {
    const root = $("v2DataRoot").value.trim();
    if (!root) { alert("评测模式请先填写数据目录"); return; }
    body.data_root = root;
  }
  const button = $("btnV2Run");
  button.disabled = true;
  try {
    const result = await apiJson("/api/v2/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // The same SSE log stream drives the floating card; on completion the
    // freshly written run is selected and its result is re-fetched.
    streamJob(result.job_id, async (ok, jobResult) => {
      button.disabled = false;
      if (!ok || !jobResult || !jobResult.run_id) return;
      await loadV2Runs();
      $("v2RunSelect").value = jobResult.run_id;
      syncM3Select($("v2RunSelect"));
      await loadV2Run(jobResult.run_id);
    });
  } catch (err) {
    button.disabled = false;
    setBadge($("runBadge"), "启动失败", "err");
    $("logWrap").classList.remove("hidden", "collapsed");
    $("runDots").classList.add("err");
    $("runTitle").textContent = "启动失败";
    $("log").classList.remove("hidden");
    $("btnRawLog").textContent = "隐藏日志";
    $("log").textContent = err.message;
  }
}

/* ------------------------------------------------------------ video crop */
function setKind(kind) {
  setV2Mode(false);
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

/* ============================================================ 交互升级 T01–T12 */
/* 约束：不改任何既有 DOM id 与后端路由；--viz-* 三种 SAR 配色一律不动。 */

const PREFS_KEY = "videosar_ui_prefs_v1";
const ZOOM_MIN = 1;
const ZOOM_MAX = 12;

const ui = {
  zoom: { s: 1, tx: 0, ty: 0 },
  panning: false,
  panStart: null,
  colSig: "",
  peekPrev: null,
  lightbox: { open: false, idx: 0 },
  theme: "light",
  split: 0.5,
  stripDrag: false,
};

const ICON_SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"></circle><path d="M12 2.6v2.4M12 19v2.4M2.6 12h2.4M19 12h2.4M5.4 5.4l1.7 1.7M16.9 16.9l1.7 1.7M18.6 5.4l-1.7 1.7M7.1 16.9l-1.7 1.7"></path></svg>';
const ICON_MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"></path></svg>';

/* ---------------------------------------------------------- 偏好持久化 (T04/T05) */
function readPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {}; } catch { return {}; }
}

function writePrefs(patch) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify({ ...readPrefs(), ...patch })); } catch { /* 隐私模式下静默降级 */ }
}

const PREF_FIELDS = ["dispMode", "dispCmap", "dispDiff", "dispDenoise", "dispTruth", "dispPred", "dispPixelated"];

function savePrefs() {
  const out = { fps: $("fps").value };
  PREF_FIELDS.forEach((id) => { out[id] = $(id).type === "checkbox" ? $(id).checked : $(id).value; });
  writePrefs(out);
}

function applyPrefs() {
  const p = readPrefs();
  PREF_FIELDS.forEach((id) => {
    if (p[id] === undefined) return;
    const el = $(id);
    if (el.type === "checkbox") el.checked = Boolean(p[id]);
    else if ([...el.options].some((o) => o.value === p[id])) el.value = p[id];
  });
  if (p.fps && [...$("fps").options].some((o) => o.value === p.fps)) $("fps").value = p.fps;
  syncM3Select($("dispMode"));
  syncM3Select($("dispCmap"));
  syncM3Select($("dispDiff"));
  syncM3Select($("fps"));
  $("columns").classList.toggle("pixelated", $("dispPixelated").checked);
  $("quad").classList.toggle("pixelated", $("dispPixelated").checked);
  setRailCollapsed(Boolean(p.railCollapsed), false);
  applyTheme(p.theme === "dark" ? "dark" : "light", false);
  syncModelChip();
}

/* ---------------------------------------------------------- T04 左侧栏收折 */
function setRailCollapsed(on, persist = true) {
  const app = document.querySelector(".app");
  if (!app) return;
  app.classList.toggle("rail-collapsed", Boolean(on));
  const btn = $("btnRailToggle");
  if (btn) btn.setAttribute("aria-expanded", String(!on));
  if (persist) writePrefs({ railCollapsed: Boolean(on) });
}

function toggleRail() {
  const app = document.querySelector(".app");
  setRailCollapsed(!app.classList.contains("rail-collapsed"));
}

/* ---------------------------------------------------------- T05 暗房模式 */
function applyTheme(theme, persist = true) {
  ui.theme = theme === "dark" ? "dark" : "light";
  if (ui.theme === "dark") document.documentElement.setAttribute("data-theme", "dark");
  else document.documentElement.removeAttribute("data-theme");
  const icon = $("themeIcon");
  if (icon) icon.innerHTML = ui.theme === "dark" ? ICON_SUN : ICON_MOON;
  const btn = $("btnTheme");
  if (btn) btn.title = ui.theme === "dark" ? "切换到浅色模式 (T)" : "切换暗房模式 (T)";
  if (persist) writePrefs({ theme: ui.theme });
  // 分段控制器滑块是按像素测量的，主题切换后要重测
  requestAnimationFrame(syncSegIndicators);
}

function toggleTheme() {
  applyTheme(ui.theme === "dark" ? "light" : "dark");
}

/* ---------------------------------------------------------- T09 分段滑块 */
function syncSegIndicator(seg) {
  if (!seg) return;
  const on = seg.querySelector("button.on");
  const ind = seg.querySelector(".seg-indicator");
  if (!on || !ind) return;
  seg.classList.add("has-indicator");
  ind.style.left = `${on.offsetLeft}px`;
  ind.style.width = `${on.offsetWidth}px`;
}

function syncSegIndicators() {
  document.querySelectorAll(".seg").forEach(syncSegIndicator);
}

function installSegIndicators() {
  document.querySelectorAll(".seg").forEach((seg) => {
    if (!seg.querySelector(".seg-indicator")) {
      const ind = document.createElement("span");
      ind.className = "seg-indicator";
      ind.setAttribute("aria-hidden", "true");
      seg.insertBefore(ind, seg.firstChild);
    }
    seg.addEventListener("click", () => requestAnimationFrame(() => syncSegIndicator(seg)));
  });
  syncSegIndicators();
}

/* ---------------------------------------------------------- T01 同步缩放 */
function zoomTransform() {
  const { s, tx, ty } = ui.zoom;
  return `translate(${(tx * 100).toFixed(4)}%, ${(ty * 100).toFixed(4)}%) scale(${s.toFixed(4)})`;
}

function applyZoom() {
  const cols = $("columns");
  if (!cols) return;
  const t = zoomTransform();
  cols.querySelectorAll(".col-img img").forEach((img) => { img.style.transform = t; });
  const label = `${ui.zoom.s.toFixed(1)}×`;
  cols.querySelectorAll(".zoom-badge").forEach((b) => {
    b.textContent = label;
    b.classList.toggle("show", ui.zoom.s > 1.001);
  });
}

function clampPan() {
  const half = Math.max(0, (ui.zoom.s - 1) / 2);
  ui.zoom.tx = Math.max(-half, Math.min(half, ui.zoom.tx));
  ui.zoom.ty = Math.max(-half, Math.min(half, ui.zoom.ty));
}

/* 以鼠标所在点为不动点缩放。位移用百分比，各列宽度不同也能对齐同一处。 */
function setZoom(nextScale, fx, fy) {
  const s0 = ui.zoom.s;
  const s1 = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, nextScale));
  if (Math.abs(s1 - s0) < 1e-6) return;
  ui.zoom.tx += (fx - 0.5) * (s0 - s1);
  ui.zoom.ty += (fy - 0.5) * (s0 - s1);
  ui.zoom.s = s1;
  if (s1 <= ZOOM_MIN + 1e-4) {
    ui.zoom.tx = 0;
    ui.zoom.ty = 0;
  }
  clampPan();
  applyZoom();
}

function resetZoom() {
  ui.zoom = { s: 1, tx: 0, ty: 0 };
  applyZoom();
}

function onColumnsWheel(ev) {
  if (state.view !== "compare" || !state.results.length) return;
  const box = ev.target.closest(".col-img");
  if (!box) return;
  ev.preventDefault();
  const r = box.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const fx = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
  const fy = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
  const cols = $("columns");
  cols.classList.add("zooming");
  setZoom(ui.zoom.s * Math.exp(-ev.deltaY * 0.0016), fx, fy);
  clearTimeout(onColumnsWheel._timer);
  onColumnsWheel._timer = setTimeout(() => cols.classList.remove("zooming"), 140);
}

function onColumnsPointerDown(ev) {
  if (state.view !== "compare") return;
  if (ui.zoom.s <= ZOOM_MIN + 1e-3 || ev.button !== 0) return;
  const box = ev.target.closest(".col-img");
  if (!box) return;
  const r = box.getBoundingClientRect();
  ui.panning = true;
  ui.panStart = { x: ev.clientX, y: ev.clientY, tx: ui.zoom.tx, ty: ui.zoom.ty, w: r.width, h: r.height };
  box.classList.add("panning");
  try { box.setPointerCapture(ev.pointerId); } catch { /* 某些浏览器不支持 */ }
  $("columns").classList.add("zooming");
  ev.preventDefault();
}

function syncCrosshair(ev) {
  const cols = $("columns");
  if (!cols || cols.classList.contains("hidden")) return;
  const boxes = cols.querySelectorAll(".col-img");
  const box = ev.target.closest ? ev.target.closest(".col-img") : null;
  if (!box) {
    boxes.forEach((b) => b.classList.remove("crosshair-on"));
    return;
  }
  const r = box.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const fx = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
  const fy = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
  boxes.forEach((b) => {
    b.classList.add("crosshair-on");
    const h = b.querySelector(".xhair-h");
    const v = b.querySelector(".xhair-v");
    if (h) h.style.top = `${(fy * 100).toFixed(3)}%`;
    if (v) v.style.left = `${(fx * 100).toFixed(3)}%`;
  });
}

function onColumnsPointerMove(ev) {
  if (!ui.panning || !ui.panStart) {
    syncCrosshair(ev);
    return;
  }
  const st = ui.panStart;
  if (!st.w || !st.h) return;
  ui.zoom.tx = st.tx + (ev.clientX - st.x) / st.w;
  ui.zoom.ty = st.ty + (ev.clientY - st.y) / st.h;
  clampPan();
  applyZoom();
}

function onColumnsPointerUp() {
  if (!ui.panning) return;
  ui.panning = false;
  ui.panStart = null;
  document.querySelectorAll("#columns .col-img.panning").forEach((b) => b.classList.remove("panning"));
  $("columns").classList.remove("zooming");
}

function clearCrosshair() {
  document.querySelectorAll("#columns .col-img.crosshair-on").forEach((b) => b.classList.remove("crosshair-on"));
}

/* ---------------------------------------------------------- T03 原图预览 + 灯箱 */
function peekBadge(show) {
  let el = $("peekBadge");
  if (!el) {
    el = document.createElement("div");
    el.id = "peekBadge";
    el.className = "peek-badge hidden";
    el.style.top = "58px";
    el.textContent = "原图预览中";
    (document.querySelector(".stage") || document.body).appendChild(el);
  }
  el.classList.toggle("hidden", !show);
}

async function setPeek(on) {
  const truth = $("dispTruth");
  const pred = $("dispPred");
  if (!truth || !pred) return;
  if (on) {
    if (ui.peekPrev) return;
    ui.peekPrev = { truth: truth.checked, pred: pred.checked };
    if (!truth.checked && !pred.checked) return;   // 已经只看原图
    truth.checked = false;
    pred.checked = false;
    peekBadge(true);
  } else {
    if (!ui.peekPrev) { peekBadge(false); return; }
    truth.checked = ui.peekPrev.truth;
    pred.checked = ui.peekPrev.pred;
    ui.peekPrev = null;
    peekBadge(false);
  }
  if (!state.results.length) return;
  if (state.view === "split") await renderSplit();
  else await render();
}

function lightboxSides() {
  return state.sides.filter((s) => s && s.image);
}

function renderLightboxBar() {
  const bar = $("lightboxBar");
  const sides = lightboxSides();
  bar.innerHTML = sides
    .map((s, i) => `<button class="lightbox-pill${i === ui.lightbox.idx ? " on" : ""}" data-lb="${i}" type="button">${i + 1} · ${esc(s.label)}</button>`)
    .join("") + `<button class="lightbox-pill" data-lb-close="1" type="button">关闭 Esc</button>`;
  bar.querySelectorAll("[data-lb]").forEach((b) => {
    b.addEventListener("click", () => {
      ui.lightbox.idx = Number(b.dataset.lb);
      renderLightboxBar();
      paintLightbox();
    });
  });
  const closer = bar.querySelector("[data-lb-close]");
  if (closer) closer.addEventListener("click", closeLightbox);
}

function paintLightbox() {
  const sides = lightboxSides();
  const cur = sides[ui.lightbox.idx];
  if (!cur) return;
  $("lightboxImg").src = cur.image;
  $("lightboxImg").alt = cur.label;
  $("lightboxCap").textContent = `${cur.label} · 帧 ${state.index + 1} / ${state.count}`;
}

function openLightbox(idx = 0) {
  const sides = lightboxSides();
  if (!sides.length) return;
  ui.lightbox.open = true;
  ui.lightbox.idx = Math.max(0, Math.min(sides.length - 1, idx));
  $("lightbox").classList.remove("hidden");
  renderLightboxBar();
  paintLightbox();
}

function closeLightbox() {
  if (!ui.lightbox.open) return;
  ui.lightbox.open = false;
  $("lightbox").classList.add("hidden");
}

/* ---------------------------------------------------------- T07 快捷键 */
function toggleKbdHelp(force) {
  const box = $("kbdHelp");
  const show = force === undefined ? box.classList.contains("hidden") : force;
  box.classList.toggle("hidden", !show);
}

function typingTarget() {
  const el = document.activeElement;
  if (!el) return false;
  if (el.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);
}

function onGlobalKeydown(ev) {
  const key = ev.key;

  if (key === "Escape") {
    if (ui.lightbox.open) { ev.preventDefault(); closeLightbox(); return; }
    if (!$("kbdHelp").classList.contains("hidden")) { ev.preventDefault(); toggleKbdHelp(false); return; }
    return;
  }
  if (typingTarget()) return;
  if (!$("metricSheet").classList.contains("hidden")) return;

  if (key === "?") { ev.preventDefault(); toggleKbdHelp(); return; }
  // v2 看板里不响应播放/步进/灯箱等影像键（速查卡除外）
  if (isV2Mode()) return;
  if (key === "h" || key === "H") { if (!ev.repeat) setPeek(true); return; }
  if (key === "t" || key === "T") { ev.preventDefault(); toggleTheme(); return; }
  if (key === "[") { ev.preventDefault(); toggleRail(); return; }
  if (key === "0") { ev.preventDefault(); resetZoom(); return; }
  if (key === "f" || key === "F") {
    ev.preventDefault();
    if (ui.lightbox.open) closeLightbox();
    else openLightbox(0);
    return;
  }
  if (["1", "2", "3"].includes(key)) {
    const idx = Number(key) - 1;
    if (idx < lightboxSides().length) { ev.preventDefault(); openLightbox(idx); }
    return;
  }

  if (!state.results.length || state.view === "video") return;

  if (key === ",") { ev.preventDefault(); gotoFrame(state.index - 1); return; }
  if (key === ".") { ev.preventDefault(); gotoFrame(state.index + 1); return; }
  if (key === "j" || key === "J") { ev.preventDefault(); gotoFrame(state.index - 1); return; }
  if (key === "l" || key === "L") { ev.preventDefault(); gotoFrame(state.index + 1); return; }
  if (key === "k" || key === "K") { ev.preventDefault(); togglePlayback(); return; }
  if (key === "Home") { ev.preventDefault(); gotoFrame(0); return; }
  if (key === "End") { ev.preventDefault(); gotoFrame(state.count - 1); return; }
}

function onGlobalKeyup(ev) {
  if (ev.key === "h" || ev.key === "H") setPeek(false);
}

/* ---------------------------------------------------------- T07 胶片条拖拽擦洗 */
function bindStripScrub() {
  const strip = $("strip");
  if (!strip) return;
  const frameUnder = (x, y) => {
    const el = document.elementFromPoint(x, y);
    const btn = el && el.closest ? el.closest("button[data-frame]") : null;
    return btn ? Number(btn.dataset.frame) : null;
  };
  strip.addEventListener("pointerdown", (ev) => {
    const btn = ev.target.closest && ev.target.closest("button[data-frame]");
    if (!btn) return;
    ui.stripDrag = true;
    try { strip.setPointerCapture(ev.pointerId); } catch { /* noop */ }
    stopPlayback();
    gotoFrame(Number(btn.dataset.frame));
    ev.preventDefault();
  });
  strip.addEventListener("pointermove", (ev) => {
    if (!ui.stripDrag) return;
    const f = frameUnder(ev.clientX, ev.clientY);
    if (f !== null && f !== state.index) gotoFrame(f);
  });
  const stop = () => { ui.stripDrag = false; };
  strip.addEventListener("pointerup", stop);
  strip.addEventListener("pointercancel", stop);
  window.addEventListener("pointerup", stop);
}

/* ---------------------------------------------------------- T08 复制与出图 */
function flashCopied(btn) {
  if (!btn) return;
  const old = btn.textContent;
  btn.classList.add("copied");
  btn.textContent = "已复制 ✓";
  clearTimeout(btn._copyTimer);
  btn._copyTimer = setTimeout(() => {
    btn.classList.remove("copied");
    btn.textContent = old;
  }, 1400);
}

function copyText(text, btn) {
  const done = () => flashCopied(btn);
  const fallback = () => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch { /* noop */ }
    ta.remove();
    done();
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(fallback);
  } else {
    fallback();
  }
}

function sidesForExport() {
  return state.sides.filter((s) => s && s.summary);
}

function metricsMarkdown() {
  const sides = state.sides;
  if (!sides.length) return "";
  const head = `| 指标 | ${sides.map((s) => s.label).join(" | ")} |`;
  const sep = `| --- | ${sides.map(() => "---:").join(" | ")} |`;
  const body = METRIC_ROWS
    .map((r) => `| ${r.label} | ${sides.map((s) => fmt(s.summary ? s.summary[r.key] : null, r.digits)).join(" | ")} |`)
    .join("\n");
  return `${head}\n${sep}\n${body}`;
}

function metricsCsv() {
  const sides = state.sides;
  const quote = (v) => {
    const t = String(v ?? "");
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
  };
  const lines = [["metric", ...sides.map((s) => quote(s.label))].join(",")];
  METRIC_ROWS.forEach((r) => {
    const cells = sides.map((s) => {
      const v = s.summary ? s.summary[r.key] : null;
      return Number.isFinite(v) ? v : "";
    });
    lines.push([quote(r.key), ...cells].join(","));
  });
  return lines.join("\n");
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片无法解码"));
    img.src = src;
  });
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function exportComparePng() {
  const sides = state.sides.filter((s) => s && s.image);
  if (!sides.length) return;
  const btn = $("btnExportPng");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "渲染中…";
  try {
    const imgs = await Promise.all(sides.map((s) => loadImage(s.image)));
    const pad = 26, gap = 18, titleH = 52, labelH = 30, metaH = 30;
    const cellW = Math.max(...imgs.map((i) => i.naturalWidth));
    const cellH = Math.max(...imgs.map((i) => i.naturalHeight));
    const w = pad * 2 + cellW * imgs.length + gap * (imgs.length - 1);
    const h = pad * 2 + titleH + labelH + cellH + metaH;
    const ratio = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const cv = document.createElement("canvas");
    cv.width = Math.round(w * ratio);
    cv.height = Math.round(h * ratio);
    const g = cv.getContext("2d");
    g.scale(ratio, ratio);

    // 固定深底浅字：导出图不随主题变化，放进论文/PPT 都稳定
    g.fillStyle = "#0f1216";
    g.fillRect(0, 0, w, h);

    const seqName = (state.seq || "").split(/[\\/]/).filter(Boolean).pop() || "序列";
    g.fillStyle = "#eef1f5";
    g.font = "600 17px system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
    g.textBaseline = "top";
    g.fillText(`${seqName} · 帧 ${state.index + 1} / ${state.count}`, pad, pad);
    g.fillStyle = "#9aa3ad";
    g.font = "12px system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
    g.fillText(`Ka 视频 SAR 阴影对比 · ${new Date().toLocaleString("zh-CN")}`, pad, pad + 24);

    // 图例：真值绿 / 预测红，与界面里的 --viz-* 完全一致
    const legend = [["真值轮廓", "#40eb78"], ["预测轮廓", "#ff5c5c"]];
    let lx = w - pad;
    g.font = "12px system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
    [...legend].reverse().forEach(([text, color]) => {
      const tw = g.measureText(text).width;
      g.fillStyle = "#cfd5dc";
      g.fillText(text, lx - tw, pad + 6);
      lx -= tw + 8;
      g.fillStyle = color;
      g.fillRect(lx - 9, pad + 8, 9, 9);
      lx -= 22;
    });

    const top = pad + titleH;
    imgs.forEach((img, i) => {
      const x = pad + i * (cellW + gap);
      g.fillStyle = "#0b0d10";
      g.fillRect(x, top, cellW, cellH);
      const scale = Math.min(cellW / img.naturalWidth, cellH / img.naturalHeight);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      g.drawImage(img, x + (cellW - dw) / 2, top + (cellH - dh) / 2, dw, dh);

      const s = sides[i].summary || {};
      g.fillStyle = "#eef1f5";
      g.font = "600 13px system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
      g.fillText(`${i + 1} · ${sides[i].label}`, x, top - labelH + 8);
      g.fillStyle = "#9aa3ad";
      g.font = "12px system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif";
      const line = `F1 ${fmt(s.target_f1)} · IoU ${fmt(s.iou)} · 虚警/帧 ${fmt(s.fp_per_frame, 2)}`;
      g.fillText(line, x, top + cellH + 8);
    });

    const blob = await new Promise((resolve) => cv.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("PNG 生成失败");
    saveBlob(blob, `sar-compare-${state.index + 1}.png`);
  } catch (err) {
    btn.textContent = "导出失败";
    setTimeout(() => { btn.textContent = old; }, 1600);
    return;
  } finally {
    btn.disabled = false;
  }
  btn.textContent = old;
}

/* ---------------------------------------------------------- T10 模型路径芯片 */
function syncModelChip() {
  const chip = $("modelChip");
  const name = $("modelChipName");
  if (!chip || !name) return;
  const full = $("modelPath").value.trim();
  const base = full ? full.split(/[\\/]/).filter(Boolean).pop() : "";
  name.textContent = base || "默认权重";
  name.classList.toggle("pc-empty", !base);
  chip.title = full || "留空用默认权重";
}

function expandModelField() {
  const input = $("modelPath");
  $("modelChip").classList.add("hidden");
  input.classList.remove("hidden");
  input.focus();
  input.select();
}

function collapseModelField() {
  const input = $("modelPath");
  if (input.classList.contains("hidden")) return;
  input.classList.add("hidden");
  $("modelChip").classList.remove("hidden");
  syncModelChip();
}

/* ---------------------------------------------------------- T11 卷帘对比 */
function ensureSplitStage() {
  let host = $("splitStage");
  if (!host) {
    host = document.createElement("div");
    host.id = "splitStage";
    host.className = "split-stage hidden";
    host.style.margin = "0 0 10px";
    const anchor = $("colHint");
    anchor.parentNode.insertBefore(host, anchor);
  }
  return host;
}

function bindSplitDrag(host) {
  const divider = host.querySelector(".split-divider");
  if (!divider) return;
  const apply = () => host.style.setProperty("--split", `${(ui.split * 100).toFixed(2)}%`);
  const setFromX = (cx) => {
    const r = host.getBoundingClientRect();
    if (!r.width) return;
    ui.split = Math.max(0.02, Math.min(0.98, (cx - r.left) / r.width));
    apply();
  };
  divider.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    try { divider.setPointerCapture(ev.pointerId); } catch { /* noop */ }
    const move = (e) => setFromX(e.clientX);
    const up = (e) => {
      divider.removeEventListener("pointermove", move);
      divider.removeEventListener("pointerup", up);
      try { divider.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    };
    divider.addEventListener("pointermove", move);
    divider.addEventListener("pointerup", up);
  });
  divider.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    ui.split = Math.max(0.02, Math.min(0.98, ui.split + (ev.key === "ArrowLeft" ? -0.02 : 0.02)));
    apply();
  });
}

async function renderSplit() {
  const filled = state.slots
    .map((idx, slot) => ({ slot, idx }))
    .filter((x) => x.idx >= 0 && state.results[x.idx]);
  const host = ensureSplitStage();
  host.classList.remove("hidden");

  if (!filled.length) {
    host.innerHTML = `<span class="lbl" style="color:#9aa3ad;">先在左侧勾选结果，再切到卷帘对比。</span>`;
    return;
  }
  let data;
  try {
    data = await fetchFrame(filled, state.index);
  } catch (err) {
    host.innerHTML = `<span class="lbl" style="color:#ff8a80;">${esc(err.message)}</span>`;
    return;
  }
  state.count = data.count;
  state.index = data.index;
  frameLabel();
  state.sides = data.sides;

  const top = data.sides[0];
  const bottom = data.sides[1] || data.sides[0];
  const rightTag = data.sides.length > 1 ? bottom.label : "仅 1 个结果 · 左右相同";
  host.innerHTML = `
    <img class="split-top" src="${top.image}" alt="${esc(top.label)}" draggable="false" />
    <img class="split-bottom" src="${bottom.image}" alt="${esc(bottom.label)}" draggable="false" />
    <span class="split-tag left">${esc(top.label)}</span>
    <span class="split-tag right">${esc(rightTag)}</span>
    <div class="split-divider" role="separator" aria-orientation="vertical" aria-label="卷帘分隔线（可拖动，方向键微调）" tabindex="0"></div>`;
  host.style.setProperty("--split", `${(ui.split * 100).toFixed(2)}%`);
  bindSplitDrag(host);
  renderPills(state.sides);
  renderMetricTable(state.sides);
  prefetchNeighbors(filled);
}

/* ---------------------------------------------------------- 统一挂载 */
function bindUpgrades() {
  installSegIndicators();

  $("btnRailToggle").addEventListener("click", toggleRail);
  $("btnRailOpen").addEventListener("click", toggleRail);
  $("btnTheme").addEventListener("click", toggleTheme);

  const modelInput = $("modelPath");
  $("modelChip").addEventListener("click", expandModelField);
  modelInput.addEventListener("focusout", collapseModelField);
  // 有些环境（无焦点窗口、iframe）不会派发 blur/focusout，点别处兜底收起
  document.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest || !ev.target.closest("#modelField")) collapseModelField();
  });
  modelInput.addEventListener("change", syncModelChip);
  modelInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); collapseModelField(); }
    if (ev.key === "Escape") { ev.preventDefault(); collapseModelField(); }
  });

  const cols = $("columns");
  cols.addEventListener("wheel", onColumnsWheel, { passive: false });
  cols.addEventListener("pointerdown", onColumnsPointerDown);
  cols.addEventListener("pointermove", onColumnsPointerMove);
  cols.addEventListener("pointerup", onColumnsPointerUp);
  cols.addEventListener("pointercancel", onColumnsPointerUp);
  cols.addEventListener("pointerleave", (ev) => {
    onColumnsPointerUp();
    if (!ev.relatedTarget || !cols.contains(ev.relatedTarget)) clearCrosshair();
  });
  cols.addEventListener("dblclick", (ev) => {
    if (ev.target.closest(".col-img")) { ev.preventDefault(); resetZoom(); }
  });
  cols.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-zoom-side]");
    if (!btn) return;
    ev.stopPropagation();
    openLightbox(Number(btn.dataset.zoomSide));
  });

  $("lightbox").addEventListener("click", (ev) => {
    if (ev.target === $("lightbox")) closeLightbox();
  });
  $("kbdHelp").addEventListener("click", (ev) => {
    if (ev.target === $("kbdHelp")) toggleKbdHelp(false);
  });

  $("btnExportPng").addEventListener("click", exportComparePng);
  $("btnCopyMd").addEventListener("click", () => {
    if (!state.sides.length) return;
    copyText(metricsMarkdown(), $("btnCopyMd"));
  });
  $("btnCopyCsv").addEventListener("click", () => {
    if (!state.sides.length) return;
    copyText(metricsCsv(), $("btnCopyCsv"));
  });

  ["fps", ...PREF_FIELDS].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", savePrefs);
  });

  bindStripScrub();

  window.addEventListener("keydown", onGlobalKeydown);
  window.addEventListener("keyup", onGlobalKeyup);
  window.addEventListener("blur", () => {
    if (ui.peekPrev) setPeek(false);
  });
  window.addEventListener("resize", () => {
    syncSegIndicators();
    applyZoom();
  });
}

/* ------------------------------------------------------------ events */
function bind() {
  upgradeAllSelects(document);
  bindUpgrades();

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
  $("v2Mode").addEventListener("change", v2ModeChanged);
  $("btnV2Refresh").addEventListener("click", () =>
    loadV2Runs().catch((err) => setBadge($("runBadge"), err.message, "err")));
  $("v2RunSelect").addEventListener("change", (e) =>
    loadV2Run(e.target.value).catch((err) => setBadge($("runBadge"), err.message, "err")));
  $("btnV2Run").addEventListener("click", runV2Benchmark);
  $("btnV2Back")?.addEventListener("click", exitV2Mode);
  $("btnRawLog").addEventListener("click", (ev) => {
    ev.stopPropagation();
    $("logWrap").classList.remove("collapsed");
    const isHidden = $("log").classList.toggle("hidden");
    $("btnRawLog").textContent = isHidden ? "原始日志" : "隐藏日志";
  });
  $("btnToggleLog").addEventListener("click", (ev) => {
    ev.stopPropagation();
    $("logWrap").classList.toggle("collapsed");
  });
  $("btnCloseLog").addEventListener("click", (ev) => {
    ev.stopPropagation();
    $("logWrap").classList.add("hidden");
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
    // v2.0 评测看板与模态弹层打开时，不允许空格/方向键在后台切换影像帧
    if (isV2Mode()) return;
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
boot()
  .then(() => {
    // 偏好项依赖 boot() 灌好的下拉选项，必须等它完成再还原
    applyPrefs();
    requestAnimationFrame(syncSegIndicators);
  })
  .catch((err) => {
    const el = $("stageEmpty");
    if (el) { el.classList.remove("hidden"); el.textContent = `初始化失败：${err.message}`; }
  });