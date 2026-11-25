# 历史模块索引（v1 一代研究 CLI）

> 状态说明，不是删除计划。本文件只做标注：**哪些代码属于仍在运行的一代研究管线，哪些属于 v2.0 主链路**。目前不移动、不删除任何文件。

## 一、两套并存的入口

| 入口 | 命令 | 评测口径 | 定位 |
| :--- | :--- | :--- | :--- |
| v1 研究 CLI | `.venv\Scripts\python.exe -m videosar --help` | 像素级 IoU、掩膜指标（96×64 图像域） | 一代实验，用于复现既有 deliverables |
| v2.0 基准 CLI | `.venv\Scripts\python.exe scripts/run_benchmark.py --help` | 目标级 FROC / mAP@0.5/0.3 / MOTA / IDF1 / HOTA | 实测盲测 + 仿真消融，主验收链路 |

**写新实验时优先用 v2.0 入口**；v1 CLI 只在需要复现旧图表或对照旧论文数字时使用。

## 二、`videosar/` 中的 v1 研究脚本

这些文件**不是死代码**：`videosar/__main__.py` 仍把它们挂在具体子命令上。删除前必须同步改 `__main__.py` 与对应 deliverables。

| 脚本 | CLI 子命令 | 被谁依赖 | 对应交付目录 |
| :--- | :--- | :--- | :--- |
| `stress.py` | `stress` | — | `deliverables/` 压测结果 |
| `echo_attribution.py` | `echo-attribution` | 被 study 链 import（`frozen_settings`/`file_hash`） | `deliverables/echo-attribution/` |
| `aperture_study.py` | `aperture-study` | `aperture_motion_study.py`、`contrast_study.py` | `deliverables/aperture-study/` |
| `aperture_motion_study.py` | `aperture-motion-study` | 引用 `aperture_study.py`、`echo_attribution.py` | `deliverables/aperture-motion-study/` |
| `contrast_study.py` | `contrast-study` | `center_study.py` | `deliverables/contrast-study/` |
| `center_box.py` / `center_study.py` | `center-study` | `open_compare.py`、`center_study.py` | `deliverables/center-study/` |
| `trajectory.py` / `trajectory_study.py` | `trajectory-study`、`trajectory-detect` | `miss_diagnostics.py` | `deliverables/trajectory-study/` |
| `small_shadow.py` | `small-shadow` | `open_compare.py`、`center_study.py`、`miss_diagnostics.py`、`echo_attribution.py` | `deliverables/small-shadow/` |
| `open_compare.py` | `open-compare` | `center_study.py`、`miss_diagnostics.py` | `deliverables/open-compare/` |
| `temporal_study.py` | `temporal-study` | `center_study.py`、`miss_diagnostics.py` | `deliverables/temporal-study/` |
| `miss_diagnostics.py` | — | 消费上面多个 study 的产物 | `deliverables/miss-diagnostics/` |
| `snl_causal.py` | — | 被 `deliverables/snl-causal-study/*.py` 中 6 个脚本 import（`CausalFeatures`、`TrackFilter`、`matched_boxes`、`box_iou`） | `deliverables/snl-causal-study/` |

**耦合特点**：上述脚本之间存在环环相扣的 import 链（`aperture_study → echo_attribution → open_compare → center_box / small_shadow → temporal_study → center_study`）。单独归档其中一个会断链，必须整组迁并修引用。

## 三、`deliverables/` 历史目录

| 目录 | 归属 | 备注 |
| :--- | :--- | :--- |
| `aperture-study/`、`aperture-motion-study/`、`center-study/`、`contrast-study/`、`echo-attribution/`、`open-compare/`、`small-shadow/`、`temporal-study/`、`trajectory-study/`、`miss-diagnostics/` | v1 study 链产物 | 由第二节脚本生成 |
| `sar90/`、`proposal/` | v1 数据集与提案实验 | 96×64 图像域口径 |
| `snl-domain-study/`、`snl-causal-study/` | SNL 迁移与因果实验 | `snl-causal-study` 含本地 YOLO 产物与权重引用 |
| `results/`、`figures/`、`models/`、`closeout/`、`report.md`、`defense.pptx` | 汇总交付物 | 结题材料，勿轻动 |

`snl-causal-study/` 额外关联 `runs/snl-causal-study/prospective/` 下的 YOLO11 系列 checkpoint/decision JSON，以及 `deliverables/snl-causal-study/yolo11n.pt`（本地权重，已被 gitignore 忽略其副本）。归档该目录须同时处理这些外部引用。

## 四、根目录快照文件

| 文件 | 说明 |
| :--- | :--- |
| `videosar-gui-all-source.txt` | GUI 源码拼接快照，便于一次性阅读 |
| `videosar-gui-code.zip` | GUI 代码打包快照 |
| `requirements-gui.txt` / `requirements-web.txt` / `requirements-unet.txt` | 分场景依赖清单，仍被对应入口使用 |

前两个是只读快照，不代表当前源码；不要从中复制代码回仓库。

## 五、v2.0 主链路（勿归入 legacy）

- `videosar/datasets/`（`base.py`、`vsmb.py`、`snl.py`）：实测数据加载与严格切分
- `videosar/metrics.py`、`experiment.py`、`tracker.py`：目标级指标、隔离评测、ByteTrack-lite
- `videosar/simulation.py`、`mwp.py`、`config.py`：高分辨率物理仿真与微波光子链路
- `videosar/detectors.py`：因果检测器注册
- `scripts/run_benchmark.py`：唯一 v2.0 基准入口

## 六、后续清理的前置条件（如需执行）

1. 整组迁移第二节脚本，同步改 `videosar/__main__.py` 的子命令与第三节 deliverables 的路径引用。
2. 处理 `snl-causal-study` 对 `snl_causal.py` 的 6 处 import 及 YOLO 产物路径。
3. 迁移后在独立 worktree 跑 `.venv\Scripts\python.exe -m unittest discover -s tests -v` 与 v2.0 冒烟评测，确认两条链路都不断。
4. 清理前先确认 `origin/feat/snl-arch-upgrade` 上的 backbone 工作已合入（该分支仍有 6 个未合入 main 的提交，依赖 snl_causal 相关代码）。

在以上条件满足前，本文件仅作导航说明。
