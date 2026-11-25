# Web 工作台交接文档 — Ka 视频 SAR 阴影对比台

> 生成时间：2026-11-25 17:20 GMT+8
> 适用对象：接手 web 工作台的人或 Agent
> 代码位置：`E:/workspace/weiboguangzi-web`（分支 `feat/fastapi-web-workbench`）
> 本地实例：`http://127.0.0.1:8000`
> 项目级交接见 `docs/HANDOFF.md`（在 `docs/handoff-2026-11-25` 分支）。**本文只讲 web 工作台**，不重复项目背景。

---

## 零、先记三条红线

| # | 红线 | 原因 |
| :-- | :--- | :--- |
| 1 | 既有 DOM id 与后端路由**不改名** | 前后端、测试、外部脚本都在按名字找（清单见第五节） |
| 2 | SAR 数据三色**不动** | `--viz-truth:#40eb78`、`--viz-pred:#ff5c5c`、`--viz-crop:#ff4040`，这是数据语义色，不跟主题走 |
| 3 | 每次改 `web/app.js` 必须把 `index.html` 里的 `?v=N` 加一 | 否则浏览器拿着旧 JS 跑新 HTML，会出现"功能突然失效"的假故障 |

---

## 一、这是什么

一句话：把一个序列的 SAR 影像和多套检测结果，做成能**逐帧对比、放大、量指标、导图**的工作台。

- 技术栈：FastAPI 后端 + **原生 JS 前端（无框架、无 CDN、完全离线）** + 单文件内联 CSS
- 全部前端代码就两个文件，没有构建步骤，没有 `node_modules`：

| 文件 | 行数 | 大小 | 职责 |
| :--- | --: | --: | :--- |
| `web/index.html` | 2295 | 79 KB | 全部结构 + **全部 CSS**（内联在 `<style>`，无外链样式表） |
| `web/app.js` | 2581 | 97 KB | 全部交互逻辑，单文件、无模块、`"use strict"` |
| `server.py` | ~1290 | — | 路由、帧渲染、两级缓存、任务流 |

一次切帧的数据流：

```
gotoFrame(i) → paintFrame()
  → renderColumns() → fetchFrame()  查 LRU 缓存（40 帧）
      未命中 → POST /api/compare/frame
  → 后端读 frames.npy + predictions.npy + masks.npy
      viz.enhance() → viz.overlay_mask() → PNG data URL
  → patchColumns() 只换 <img src> 与指标文字，不重建 DOM
```

---

## 二、怎么跑起来

```bash
cd E:\workspace\weiboguangzi-web
E:\workspace\weiboguangzi\.venv\Scripts\python.exe server.py
# → http://127.0.0.1:8000
```

- 端口写死在 `server.py` 末尾：`uvicorn.run(app, host="127.0.0.1", port=8000)`
- 依赖见 `requirements-web.txt`（fastapi / uvicorn / numpy / pillow / imageio）

**关键：`web/` 之外的 `runs/` 必须有。** 这个 worktree 里 `runs` 是目录联接（junction），指向主工作区的 `E:\workspace\weiboguangzi\runs`，不复制数据：

```powershell
New-Item -ItemType Junction -Path "E:\workspace\weiboguangzi-web\runs" `
  -Target "E:\workspace\weiboguangzi\runs"
```

没有它 → `/api/bootstrap` 返回 `runs_exists:false`，左侧序列下拉是空的，页面看起来"什么都没加载"，很容易误判成前端坏了。

起完先验一次再开浏览器：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/   # 期望 200
```

---

## 三、文件里的地图

### 3.1 `index.html` 的 CSS 分区

| 行区间 | 内容 |
| :--- | :--- |
| 1–114 | `:root` 设计 token（M3 Tonal Blue）、reset |
| 115–124 | 主布局 `.app` 两列 grid |
| 125–728 | 左侧栏、M3 自定义下拉、按钮体系、折叠面板 |
| 729–1035 | 主舞台、工具栏、对比列 |
| 1036–1186 | 四宫格、指标条、胶片条 |
| 1187–1509 | 悬浮日志卡、显示 Popover、指标模态、视频区 |
| 1520–1949 | v2.0 评测看板 + **T01–T12 全部新增样式** |
| 1950–末尾 | 响应式断点 → HTML 结构 → 浮层容器（灯箱 / 快捷键卡） |

T01–T12 的样式都带 `/* ---- Txx ... ---- */` 注释，可直接跳：

```
1626 T05 暗房模式   1666 T04 侧栏收折   1680 T01 同步缩放
1730 T03 灯箱       1777 T07 快捷键     1810 T06 最优高亮
1815 T08 复制导出   1818 T09 滑块骨架   1847 T10 路径芯片
1868 T11 卷帘       1929 T12 胶片热条
```

### 3.2 `app.js` 功能 → 函数 → 行号

| 能力 | 函数 | 行 |
| :--- | :--- | --: |
| 帧缓存 LRU / 预热 | `cacheGet` `cacheSet` `fetchFrame` `prefetchNeighbors` | 531–586 |
| 列结构重建 / 原地补丁 | `rebuildColumns` `patchColumns` | 608–707 |
| 渲染主链路 | `renderColumns` `renderQuad` `renderStrip` `renderPills` | 709–965 |
| 最优高亮 | `bestIndices` `bestMap` `markBestColumns` `renderMetricTable` | 794–864 |
| v2 进出 | `isV2Mode` `setV2Mode` `exitV2Mode` | 1247–1275 |
| FROC 绘制 | `renderFroc` | 1341 |
| 偏好持久化 | `readPrefs` `writePrefs` `applyPrefs` `savePrefs` | 1536–1590 |
| 侧栏 / 主题 | `setRailCollapsed` `toggleRail` `applyTheme` `toggleTheme` | 1593–1624 |
| 分段滑块 | `syncSegIndicator` `installSegIndicators` | 1626–1655 |
| 同步缩放 | `zoomTransform` `applyZoom` `clampPan` `setZoom` `resetZoom` | 1655–1696 |
| 十字准星 / 平移 | `onColumnsWheel` `onColumnsPointerDown/Move/Up` `syncCrosshair` `clearCrosshair` | 1698–1775 |
| 原图预览 / 灯箱 | `setPeek` `paintLightbox` `openLightbox` `closeLightbox` | 1789–1859 |
| 全局键盘 | `typingTarget` `onGlobalKeydown` `onGlobalKeyup` | 1865–1916 |
| 胶片条擦洗 | `bindStripScrub` | 1918 |
| 复制 / 导 PNG | `copyText` `metricsMarkdown` `metricsCsv` `exportComparePng` | 1959–2105 |
| 路径芯片 | `syncModelChip` `expandModelField` `collapseModelField` | 2113–2135 |
| 卷帘 | `bindSplitDrag` `renderSplit` | 2154–2221 |
| 统一挂载 | `bindUpgrades` | 2226 |

### 3.3 `server.py` 里与工作台相关的路由

| 路由 | 作用 | 备注 |
| :--- | :--- | :--- |
| `GET /` | 返回 `web/index.html` | — |
| `GET /api/bootstrap` | 运行根、模式、色图、默认权重 | 前端 `boot()` 第一件事 |
| `GET /api/sequences` · `GET /api/results` | 序列清单 / 某序列下已有结果 | 左侧栏数据源 |
| `POST /api/browser/meta` | 帧数、尺寸等 | — |
| `POST /api/compare/frame` | **并排视图主力**：多列 + 指标摘要 | **尊重 `show_pred`/`show_truth`** |
| `POST /api/browser/frame` · `/frames` | 四宫格单帧 / 多帧 | — |
| `POST /api/browser/strip` | 胶片条缩略图 | 额外返回 `frame_status[]` |
| `POST /api/browser/export` | 导出 GIF/MP4 | — |
| `GET /api/v2/runs` · `/api/v2/runs/{id}` · `POST /api/v2/run` | v2.0 基准数据 | FROC 数据源 |
| `GET /api/jobs/{id}/stream` | SSE 任务流 | 日志卡与步骤条 |

`DisplaySettings`（所有渲染接口共用的设置对象，字段名不能改）：

```python
mode: str = "percentile"      # 灰度映射
cmap: str = "gray"            # 伪彩色
denoise: bool = False
diff_ref: str = "median"      # median | previous
show_truth: bool = True
show_pred: bool = True
```

---

## 四、能力清单

### 4.1 T01–T12

| 编号 | 能力 | 怎么用 | 持久化 |
| :-- | :--- | :--- | :--- |
| T01 | 三栏同步缩放 1×–12×，以鼠标为不动点；拖动平移 | 滚轮 / 拖拽 / 双击或 `0` 复位 | 不持久（切序列复位） |
| T02 | 40 帧 LRU 缓存 + 空闲预热相邻帧；换帧只改 `src` | 自动 | — |
| T03 | 按住 `H` 隐轮廓看原图；灯箱放大 | `H` / 栏头 ⤢ / `F`；`1·2·3` 切模型，`Esc` 退 | — |
| T04 | 左侧栏收折 | 栏头按钮 / `[` | ✅ |
| T05 | 暗房模式 | 工具栏月亮按钮 / `T` | ✅ |
| T06 | 最优指标高亮（≥2 列，全 10 行） | 自动 | — |
| T07 | 胶片条拖拽擦洗；快捷键 | 按住胶片条拖 / `,` `.` `Home` `End` `J` `K` `L` / `?` | — |
| T08 | 复制 Markdown / CSV；导出对比图 PNG | 指标表内两个按钮 / 工具栏「保存对比图」 | — |
| T09 | 分段滑块按像素实测；慢图骨架屏 | 自动 | — |
| T10 | 模型路径只读芯片，点击展开 | 芯片 → 输入 → 失焦/回车收起 | 不持久（用后端默认值） |
| T11 | 卷帘对比 | 视图切「卷帘」，拖分隔线 / 方向键微调 | 不持久 |
| T12 | 胶片条 3px 质量微条 | 悬停看帧号与状态 | — |

`localStorage` 键：`videosar_ui_prefs_v1`，存 `fps` / `dispMode` / `dispCmap` / `dispDiff` / `dispDenoise` / `dispTruth` / `dispPred` / `dispPixelated` / `railCollapsed` / `theme`。

### 4.2 v2.0 评测看板

左侧栏「v2.0 评测运行」是**默认折叠**的 `<details>`。跑一次基准或选一个历史运行后，主舞台切到 v2 模式：实测盲测区与仿真消融区**硬隔离两张卡**，下方 FROC 曲线（1.0 FA 蓝色虚线 + 可悬停节点）。`← 返回影像对比` 或 `Esc` 退出。

### 4.3 快捷键全表

| 键 | 作用 |
| :--- | :--- |
| `←` `→` | 上一帧 / 下一帧 |
| `空格` | 播放 / 暂停 |
| `,` `.` | 逐帧步进 |
| `J` `K` `L` | 后退 / 播放暂停 / 前进 |
| `Home` `End` | 首帧 / 末帧 |
| `1` `2` `3` | 切第 N 个模型并进灯箱 |
| `H` | 按住看原图 |
| `F` | 灯箱开 / 关 |
| `0` | 复位缩放 |
| `[` | 收起 / 展开侧栏 |
| `T` | 明暗主题 |
| `?` | 快捷键卡 |
| `Esc` | 关最上层浮层 |

---

## 五、契约红线（改名就会炸）

### 5.1 DOM id

已核对：JS 引用 **106 个 id**，HTML 里**全部存在，无孤儿引用**。其中只有 `splitStage`、`peekBadge` 是运行时创建的。

**核心工作流 id（绝对不能动）**

```
seqSelect  seqPath  kindSeg  videoFile  methodList  btnRun  modelPath
columns  quad  quadBar  quadSource  pills  strip  colHint  stageEmpty
logWrap  steps  log  runDots  runTitle  btnRawLog  btnToggleLog  btnCloseLog
metricSheet  metricTable  btnAllMetrics  btnCloseSheet
frameLabel  btnPlay  playIcon  playLabel  btnPrev  btnNext  viewSeg
btnDisplay  displayPop  dispMode  dispCmap  dispDiff  dispDenoise
dispTruth  dispPred  dispPixelated  fps  ctxChip  ctxText
```

**T01–T12 新增 id**

```
btnRailToggle  btnRailOpen  btnTheme  themeIcon  modelChip  modelChipName
btnExportPng  btnCopyMd  btnCopyCsv  lightbox  lightboxImg  lightboxBar
lightboxCap  kbdHelp  （运行时：splitStage  peekBadge）
```

**v2 看板 id**

```
v2Section  v2Mode  v2Mwp  v2MwpRow  v2DataRoot  v2DataRootField  v2RunSelect
btnV2Refresh  btnV2Run  btnV2Back  v2Panel  v2RunId  v2Meta
v2RealCard  v2RealBody  v2RealCount  v2SynthCard  v2SynthBody  v2SynthCount
v2Froc  v2FrocTip  v2FrocLegend
```

### 5.2 不能动的接口约定

- 路由路径（第三节表格）、`DisplaySettings` 字段名
- `POST /api/compare/frame` 的返回结构：`{ index, count, sides:[{ label, path, image, components, overlap, summary }] }`
- `POST /api/browser/strip` 的 `frame_status` 取值只能是 `ok` / `missed` / `fp_high` / `unknown`，前端按这四个值上色，多一个值就画不出条

### 5.3 不能动的颜色

```css
--viz-truth: #40eb78;   /* 真值轮廓，绿 */
--viz-pred:  #ff5c5c;   /* 预测轮廓，红 */
--viz-crop:  #ff4040;   /* 裁剪框，红 */
```

暗色主题只覆盖 UI token，**这三行不在覆盖范围内**。`V2_FROC_COLORS` 那组 8 色分类调色板同理，是有意硬编码的，不要改成 CSS 变量（会让多条曲线撞色）。

### 5.4 缓存号

`web/index.html` 末尾：

```html
<script src="/static/app.js?v=11"></script>
```

改 `app.js` 必 +1。当前 `v=11`。

---

## 六、验收方法

### 6.1 必做（30 秒）

```bash
node --check web/app.js      # 语法
python -c "import ast;ast.parse(open('server.py',encoding='utf-8').read())"
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
```

### 6.2 无头验收模板

项目没有前端自动化测试，实测靠 Chrome DevTools Protocol。**两个必踩的坑已经写进模板里**：视口覆盖、轮询等待。

```js
// _t.mjs —— node _t.mjs（Node 22 自带 WebSocket，无需装包）
const BASE = "http://127.0.0.1:9341";
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let ws, id = 0; const pending = new Map();
const send = (method, params = {}) => {
  const i = ++id;
  ws.send(JSON.stringify({ id: i, method, params }));
  return new Promise((res, rej) => { pending.set(i, { res, rej });
    setTimeout(() => rej(new Error("timeout " + method)), 20000); });
};
const ev = async (expr) => {
  const r = await send("Runtime.evaluate", {
    expression: `(async()=>{try{return JSON.stringify(await (${expr}))}catch(e){return JSON.stringify({__err:String(e)})}})()`,
    awaitPromise: true, returnByValue: true });
  return JSON.parse(r.result.value);
};

const list = await (await fetch(`${BASE}/json/list`)).json();
ws = new WebSocket(list.find(t => t.type === "page").webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
ws.onmessage = (m) => { const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id); msg.error ? rej(new Error(msg.error.message)) : res(msg.result); } };

await send("Runtime.enable");
await send("Page.enable");
// ① 必须覆盖视口，否则 800×600 掉进 @media(max-width:860px)
await send("Emulation.setDeviceMetricsOverride", { width: 1500, height: 940, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: "http://127.0.0.1:8000/?t=" + Date.now() });

// ② 轮询等初始化完成，不要固定 sleep
for (let i = 0; i < 40; i++) {
  await sleep(500);
  if (await ev(`document.getElementById('seqSelect').options.length > 1`) === true) break;
}
console.log(await ev(`({ cols: document.querySelectorAll('#columns .col').length, ver: document.querySelector('script[src*=app]').getAttribute('src') })`));
ws.close(); process.exit(0);
```

启动 Chrome：

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new \
  --disable-gpu --no-sandbox --remote-debugging-port=9341 \
  --remote-debugging-address=127.0.0.1 \
  --user-data-dir="E:/workspace/weiboguangzi-web/_cdp" about:blank
```

`--remote-debugging-address=127.0.0.1` 不能省：默认只绑 IPv6，`curl http://127.0.0.1:port/json/list` 会连不上。

### 6.3 可直接抄的断言

```js
// 缩放：滚轮后倍率与位移
box.dispatchEvent(new WheelEvent('wheel', { deltaY: -420, clientX: r.left + r.width*0.25, clientY: r.top + r.height*0.3, bubbles: true, cancelable: true }));
// → ui.zoom.s ≈ 1.958，tx = (0.25-0.5)*(1-1.958) ≈ 0.2395

// 缓存真的被清：塞哨兵键再换序列
colCache.set('__sentinel__', 1); await loadSequence(state.seq); !colCache.has('__sentinel__')

// 10 行最优：造两个 summary 差异明显的 side
renderMetricTable([a, b]);
document.querySelectorAll('#metricTable td.is-best').length   // 应为 10，每行 1 个

// 灯箱跟随
await gotoFrame(3); openLightbox(0); await gotoFrame(4);      // #lightboxImg.src 应变化

// FROC 主题色：切主题后要等过渡结束再读
applyTheme('dark', false); await sleep(700);
getComputedStyle(document.querySelector('#v2Froc line[stroke]')).stroke  // rgb(44,50,61)
```

---

## 七、踩过的坑（都会浪费时间）

1. **无焦点窗口不派发 `blur` / `focusout`**。无头 Chrome 里 `input.blur()` 会让 `activeElement` 变 body，但事件不触发。所以路径芯片收起用的是 `focusout` + `document.pointerdown` 双保险，别改回只监听 `blur`。

2. **切主题后立刻读颜色会读到过渡中间值**。`.btn` 上有 `transition: ... color var(--ease)`，`getComputedStyle` 在 t≈0 返回旧色（白色），会误判"CSS 没生效"。**读颜色前等 ≥600ms**。

3. **无头默认窗口 800×600 会命中 `@media (max-width:860px)`**，`.app` 变 `display:block`，`.col-img` 高度塌成 0，缩放/准星全部被"退化尺寸"守卫跳过。必须先 `Emulation.setDeviceMetricsOverride`。

4. **固定 `sleep` 等初始化不可靠**。7s 常不够，会测到 boot 半途：`#columns` 为空、模型芯片显示「默认权重」。用轮询断言替代。

5. **`--dump-dom` / CDP 首次导航要 ≥3.5s 或加 `?t=Date.now()`**，否则拿到的是空壳，误判"未初始化"。

6. **仓库 `core.autocrlf=true`**。工作区是 CRLF、blob 是 LF，所以 `diff` 会整文件全红。用 `diff --strip-trailing-cr`；改 `?v=` 用 `sed -i` 时注意会留下 LF 尾巴（对渲染无影响，`git status` 仍干净）。

7. **主工作区可能被并发进程切分支或 `reset --hard`**。历史事故：v7 的暂存改动被 `reset --hard HEAD` 清空（reflog 可见）。所以 web 工作都在独立 worktree 做，且**`cp` + `git add` + `commit` + `push` 要写在一条命令里**。

8. **SVG 呈现属性写 `fill="var(--x)"` 在 Chrome 157 可用**（已实测），但老浏览器不稳。要兼容更宽就写 `style="fill:var(--x)"`，效果等价。

9. **`.split-stage img` 必须 `inset:0; width/height:100%; object-fit:contain`**。只写 `max-width/max-height` 会按 PNG 原始小尺寸渲染（图缩成一小块）。同理 `.col-img` 需要父级有确定高度，否则图片高度为 0。

10. **`requestIdleCallback` 回退**。Safari 与部分 WebView 没有，`prefetchNeighbors` 里有 `setTimeout` 兜底，别删。

---

## 八、当前状态

| 项 | 值 |
| :--- | :--- |
| 分支 | `feat/fastapi-web-workbench`（worktree `E:/workspace/weiboguangzi-web`） |
| 平行分支 | `feat/ui-master-t12`（`E:/workspace/weiboguangzi-ui`），与本分支同点，内容逐字节一致 |
| 最新提交 | `414a909` fix(web): 暗房对比度 + 帧缓存失效/重叠冻结/全量最优/灯箱跟随/FROC 主题色 |
| 是否已推 | ✅ 两分支均已 push |
| 与 main 关系 | main（`0f1e635`）尚未包含 web 改动，需要时再合 |
| 脚本版本 | `?v=11` |
| 前端自动化测试 | ❌ 无。只有第 6.2/6.3 的手工 CDP 断言 |
| 已知未做 | 移动端触屏手势（缩放只做了鼠标）、导出图标的本地化、v2 看板的空态插画 |

**注意**：项目级 `docs/HANDOFF.md` 里关于 web 的描述已经过时（写着 `feat/fastapi-web-workbench` HEAD = `bf8ba5e`、远端已删除）。实际已重建并推进到 `414a909`，以本文为准。

---

## 九、改代码的推荐流程

```bash
# 1. 只在这一个 worktree 里改，别在主工作区动 web/
cd E:/workspace/weiboguangzi-web

# 2. 改完先过语法
node --check web/app.js

# 3. 改过 app.js 就把缓存号加一
sed -i 's|/static/app.js?v=11|/static/app.js?v=12|' web/index.html

# 4. 起服务 + 无头验收（见第 6 节）
python server.py &   # 或另开窗口

# 5. 一条命令提交并推送（防并发 reset 清暂存）
git add web/app.js web/index.html && \
  git commit -m "feat(web): ..." && \
  git push origin feat/fastapi-web-workbench

# 6. 同步平行分支
cd E:/workspace/weiboguangzi-ui && git merge --ff-only feat/fastapi-web-workbench && git push origin feat/ui-master-t12
```

新增能力的检查清单：加结构 → 加样式（带 `/* ---- Txx ---- */` 注释）→ 加逻辑（函数放进对应分区）→ **给新 id 登记到第 5.1 节** → `node --check` → CDP 断言 → 升 `?v=` → 提交。
