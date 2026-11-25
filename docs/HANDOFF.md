# 项目交接文档 — Video-SAR 动目标检测与跟踪

> 生成时间：2026-11-25 13:27 GMT+8
> 适用对象：接手本项目的协作者或 Agent
> 仓库：`https://github.com/ame858/weiboguangzi.git`

---

## 一、项目目标与当前范式

微波光子 Ka 频段视频 SAR 动目标阴影建模与智能检测。**v2.0 的核心转向是"倒置数据主次"**：

- 以公开实测集（VSMB / SNL Eubank Gate）作为**不可作弊的盲测主门槛**
- 物理仿真与微波光子（MWP）链路退化收束为**独立可控的消融支线**
- 严禁把 `real` 与 `synthetic` 指标混池求平均

当前仓库里 v1 一代 CLI（`python -m videosar`）与 v2.0 基准 CLI（`python scripts/run_benchmark.py`）**并存**。新实验一律走 v2.0 入口。历史模块清单见 `docs/legacy-module-index.md`。

---

## 二、分支与工作区全景（关键，别踩坑）

主工作区 `E:/workspace/weiboguangzi` 当前在 `main`，**有并发会话的未提交删除**（`任务大纲.md` 等 3 个教学文档），接手后不要在主工作区做 `checkout -b` / `reset --hard`。所有新工作请新建独立 worktree。

### 远端分支

| 分支 | 状态 |
| :--- | :--- |
| `origin/main` | 停在中转提交 `df48661`，本地 main 领先 5 个提交**尚未 push** |
| `origin/feat/snl-arch-upgrade` | 仍有 6 个提交未合入 main（SegFormer / ResNet34-Transformer backbone 等），**不可删** |

`origin/feat/fastapi-web-workbench` 已确认完全合入并于 2026-11-25 删除。

### 本地分支与工作区

| 分支 | HEAD | worktree | 是否合入 main | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `main` | `589e1ca` | `E:/workspace/weiboguangzi` | — | 主线，ahead 5 未 push；工作区 dirty（他人所为） |
| `refactor/data-first` | `c325e18` | `E:/workspace/weiboguangzi-refactor` | ✅ 已合入 | v2.0 数据层底座（提交 `0301e77`→`8693ffb`→`c325e18`） |
| `feat/v2-detector-registry-and-baseline` | `291a425` | `E:/workspace/weiboguangzi-detector` | ✅ 已合入 | 因果检测器注册与基线 |
| `feat/v2-web-ui-contract` | `6851c44` | `E:/workspace/weiboguangzi-webui` | ✅ 已合入 | FastAPI 对接 v2.0 四文件协议 |
| `feat/fastapi-web-workbench` | `bf8ba5e` | `E:/workspace/weiboguangzi-web` | ✅ 已合入 | 远端已删，本地仍在 |
| `feat/proposal-dataset-v2` | `d5e40cf` | `E:/workspace/weiboguangzi-proposal` | ✅ 已合入 main（2026-11-25 14:26） | **工单 4 成果**：申报书数据集 + 顶刊评测 |
| `docs/legacy-module-index` | `8c93db9` | `E:/workspace/weiboguangzi-legacy-index` | ✅ 随 proposal 分支合入 | 历史模块导航 |
| `docs/handoff-2026-11-25` | `0e247b5` | `E:/workspace/weiboguangzi-handoff` | ❌ 未合入 | 本交接文档 |
| `feat/snl-arch-upgrade` | `98341a9` | — | ❌ 6 提交未合入 | backbone 升级，YOLO11-P2 的基础 |
| `feat/snl-yolo11-p2` | `c9eb1fa` | — | ❌ 未合入 | 因果 YOLO11-P2 管线 |
| `gui` / `workbuddy/*` | — | WorkBuddy 内置 worktree | — | GUI 与快照分支，非主线 |

**合入状态**：main 已含 v2.0 全部主线成果（含工单 4），本地领先 origin 2 提交待 push。下一步再评估 `feat/snl-arch-upgrade`。

---

## 三、已完成的工作（按时间）

### 1. v2.0 数据层重构（`refactor/data-first`，已合入 main）
- `proposal_data.py` 混测捷径作废（`LegacyProposalDataRemovedError`）
- `metrics.py` / `experiment.py` 改目标级 FROC、Recall@FA、mAP@0.5/0.3、MOTA/IDF1/HOTA，修复空场景 NaN；硬隔离 real/synthetic
- 新增 `datasets/base.py`（因果滑窗 + 512×512 切块 + 序列 ID 切分）、`vsmb.py`、`snl.py`
- `config.py` 分辨率下限 256×256；`simulation.py` 多目标曲线轨迹 + 静态遮挡；`mwp.py` 默认关闭
- 归档失效旧测试到 `legacy/tests/`；`eval-vsmb` / `eval-snl` CLI 补全为完整评测

### 2. v2.0 检测器与 Web 协议（已合入 main）
- `videosar/detectors.py`：因果检测器注册（`get_detector` / `list_detectors`）
- FastAPI 工作台按 v2.0 四文件协议读取 `config.json` / `metrics.json` / `summary.csv` / `froc_curve.csv`，分区展示 FROC

### 3. 工程清理（2026-11-25）
- 删除远端已合入分支 `feat/fastapi-web-workbench`
- 新增 `docs/legacy-module-index.md` 历史模块导航，不移动文件

### 4. 申报书数据集与顶刊评测（`feat/proposal-dataset-v2`，**未合入**）
提交 `d5e40cf`，8 文件 +661/−10：
- **A** `simulation.py`：速度-阴影耦合、`enable_stop_and_go` 停启轨迹、`shadow_masks` / `static_distractor_mask` / `kinematics` 三件套；`export_mot_directory` 输出 6 类文件
- **B** `scripts/build_proposal_dataset.py`：6 分区 120 组×25 帧=3000 帧；`--smoke-scale` 12 组×8 帧；SNL 底图严格只读 ≤600 帧；`train_norm_stats` 仅来自 `id_train`
- **C** 双传统基线（单帧低灰度阈值 + 时序背景差分，均支持掩膜返回）；像素级 F1/IoU、运动学速度/航向反演指标（空场景零 NaN）；`evaluate_split` 扩展 condition_tag / 耗时 / 像素 / 运动学字段
- **D** 5 项新单测；快照 zip/txt 加 gitignore

---

## 四、数据资产现状（**关键限制**）

| 数据集 | 路径 | 状态 |
| :--- | :--- | :--- |
| SNL Eubank Gate | `E:/workspace/weiboguangzi-refactor/data/snl_v2/frames/` | 899 帧（`frame_00001.png`–`frame_00899.png`），已有 `gt.txt` |
| VSMB | — | **本地无原始数据**，全盘未找到；`outputs/vsmb_v2_baseline` 是空目录 |
| proposal_v2（生成集） | `E:/workspace/weiboguangzi-proposal/data/proposal_v2/` | 2026-11-25 全量生成：120 序列 / 3000 帧 / 199MB，已被 gitignore 排除 |

### 全量数据集已生成（2026-11-25，可复现）

```powershell
python scripts/build_proposal_dataset.py `
  --snl-train-frames-dir E:/workspace/weiboguangzi-refactor/data/snl_v2 `
  --output-dir data/proposal_v2
```

`dataset_manifest.json` 实测关键字段：
```
total_sequences = 120, total_frames = 3000
real_bg_used = true, snl_max_train_frame_used = 600
splits = {id_train:54, id_val:18, id_test:18, ood_speed:10, ood_background:10, ood_mwp_noise:10}
120 个 sequence_id 全唯一；真实 SNL 底图仅用于 id_train（39 组），其余 5 个分区为 0
train_norm_stats = {mean:95.80, std:35.93, p01:34, p99:200}（受真实底图影响，与 smoke 的 130.47 不同，属正常）
```

### 已知不可用的跑分（**根因已定位：缺测试区真值**）

`outputs/v2_eval_snl_baseline/metrics.json` 与 `outputs/snl_v2_baseline/metrics.json` 虽有 `real_blind_eval` 分区，但：

```
num_gt = 0, num_pred = 13973, map50 = 0.0, FA/frame = 49.37, empty_scene = True
```

**根因（已查清，非代码 bug）**：`data/snl_v2/gt.txt` 共 3216 行标注，但 frame_id 只覆盖 **150–527**（369 个唯一帧），而标准 SNL 测试区是 **617–900**，两者完全不重叠——测试区标注行数为 0。全仓仅此一份 SNL gt；`runs/snl-eubank-check/community-annotations.zip` 是 2020 年的分割掩膜包，不含 617+ 框标注。

**结论**：Sim-to-Real 增益对照（617–900 盲测）目前**无法产出有效指标**。需要补充覆盖 617–900 的测试区框标注，或确认现有标注可平移，否则不要硬跑、更不要写进报告。

---

## 五、如何运行（Windows PowerShell，仓库根目录）

```powershell
# 用项目虚拟环境
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# v2.0 冒烟评测（合成，generate 标准 4 文件）
.\.venv\Scripts\python.exe scripts/run_benchmark.py --mode smoke --output-dir outputs/smoke_baseline

# 申报书数据集（smoke 12 组 / 全量 120 组）
.\.venv\Scripts\python.exe scripts/build_proposal_dataset.py --smoke-scale --output-dir outputs/proposal_smoke
.\.venv\Scripts\python.exe scripts/build_proposal_dataset.py --output-dir data/proposal_v2

# SNL 真实盲测（需先确认 GT 对齐问题）
.\.venv\Scripts\python.exe scripts/run_benchmark.py --mode eval-snl --data-root data/snl_v2 --output-dir outputs/snl_blind
```

依赖安装必须用清华源：
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验收基线**：45 项测试全部通过（1 skipped）；smoke mAP@0.5 ≈ 0.029（合成，非实测）。

---

## 六、红线（务必遵守）

1. 不碰主工作区 `E:/workspace/weiboguangzi` 的未提交文件（并发会话所有）。
2. 不把 `real` 与 `synthetic` 混池；不重新启用 `proposal_data.py` 混测逻辑；分辨率不得低于 256×256。
3. SNL 真实无车底图**只允许读前 600 帧**（1–600 训练集），绝不读 601–900。
4. 归一化统计量（mean/std/p01/p99）**只允许从 `id_train` 计算**。
5. 像素级指标在空场景必须返回有限值（当前实现返回 1.0），禁止 NaN/Inf。
6. `feat/snl-arch-upgrade` 不可删（6 提交未合入，是 YOLO11-P2 基础）。
7. pip 一律加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 七、待办与优先级

| 优先级 | 事项 | 阻塞 / 备注 |
| :--- | :--- | :--- |
| P0 | 获取覆盖 SNL 617–900 的测试区框标注 | **根因已定位**：现有 gt 只到 527；阻塞全部真实盲测与 Sim-to-Real 对照 |
| P0 | 获取 VSMB 原始数据（45 序列 / 1024×1024） | 本地缺失，主基准无法跑 |
| P1 | push main 的 2 个本地领先提交到 origin | 已合入工单 4，待 push |
| P1 | 主工作区 3 个教学文档的删除：确认是并发会话有意删除还是误操作 | 未经确认不要恢复（合并时已用 stash 完整保护） |
| P2 | 评估 `feat/snl-arch-upgrade` 的 backbone 工作是否合入 | 6 提交，YOLO11-P2 依赖 |
| P2 | 合并 `docs/handoff-2026-11-25` 交接文档 | 1 提交 |
| P3 | 历史脚本/目录归档到 `legacy/` | 耦合深，见 legacy-module-index 前置条件 |

### 2026-11-25 已完成（从原待办移除）

- ✅ `feat/proposal-dataset-v2` 合入 main（`d5e40cf`，经 stash 保护并发状态后快进）
- ✅ `docs/legacy-module-index` 随 proposal 分支合入
- ✅ 全量生成 120 组 / 3000 帧数据集（199MB，manifest 已核对，真实底图不越界）
- ✅ SNL `num_gt=0` 根因查清：GT 只覆盖 150–527，测试区 617–900 无标注

---

## 八、文档与产物位置

- 方案说明：`docs/legacy-module-index.md`（本分支）
- 工作记录：`E:/workspace/weiboguangzi/.workbuddy/memory/2026-11-24.md`、`2026-11-25.md`
- 标准评测产物：`config.json` / `metrics.json`（含 `real_blind_eval` + `synthetic_ablation`）/ `summary.csv` / `froc_curve.csv`
- 申报书数据集清单：`outputs/proposal_smoke/dataset_manifest.json`（smoke 样例）

---

**一句话总结**：v2.0 底座、工单 4 数据集工具链和 120 组 / 3000 帧全量生成都已就位并合入 main，但**实测盲测结论仍未建立**——SNL 标注只到 527 帧（测试区 617–900 无 GT）、VSMB 原始数据缺失。接手第一件事应该是补齐测试区标注，而不是继续加模型。
