# Video-SAR Moving Target Detection & Tracking (v2.0 Refactor)

本仓库是 `ame858/weiboguangzi` 的 v2.0 重构版，核心目标是**倒置数据主次**：以公开实测 Video-SAR 数据集（[VSMB](https://github.com/softwarePupil/VSMB) 与 [SNL Eubank Gate](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4)）作为不可作弊的盲测验收标尺，将物理仿真与微波光子链路退化收束为独立可控的消融与增强支线。

## 一、四步重构核心变更

1. **切除混测水分与像素级缺陷指标**
   - 正式下线 `videosar/proposal_data.py` 中的无噪声二维掩膜（`source=geometry`）混测捷径，调用时直接抛出 `LegacyProposalDataRemovedError`。
   - `videosar/metrics.py` 与 `videosar/experiment.py` 全面切换至目标级 **FROC / Recall @ FA/frame**、**mAP@0.5**（及小目标 **mAP@0.3**）与多目标跟踪指标（**MOTA / IDF1 / HOTA / ID Switches**），彻底修复空场景 `union_pixels=0` 导致的 NaN 问题。
   - `ExperimentRunner` 内置硬性隔离校验：禁止将 `real` 与 `synthetic` 切分混入同一汇总均值。

2. **接入公开实测基准（VSMB + SNL）**
   - `videosar/datasets/vsmb.py`：兼容 [VSMB (arXiv:2506.12105)](https://arxiv.org/html/2506.12105) 45 序列 / 1024×1024 DarkLabel MOT 标准标注（`car`, `ship`, `airplane`），强制按 **序列 ID** 划分 `train/val/test`，杜绝同视频抽帧泄露。
   - `videosar/datasets/snl.py`：支持 [SNL Eubank Gate](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4) MP4 拆帧与标准时间块切分（前 600 帧训练、后 300 帧测试，并保留默认 16 帧时间隔离带防止滑窗泄露）。
   - `videosar/datasets/base.py`：统一提供因果滑窗（默认 `W=5`）、`512×512` 重叠切块（`overlap=64`）与精确边界框坐标重映射，原生兼容 PyTorch `Dataset` 且支持纯 NumPy 轻量运行。

3. **升级物理仿真引擎与明确“微波光子”定位**
   - `videosar/config.py`：将默认分辨率由 `96×64` 强制提升至 `≥256×256`（低于 256 直接校验报错）。
   - `videosar/simulation.py`：升级为多目标（3~12 辆）弯曲道路加减速轨迹 + 静态建筑遮挡阴影干扰物 + 真实底图物理阴影与方位向多普勒拖尾注入，并提供 [RadarSimPy](https://github.com/radarsimx/radarsimpy) 可选后端接口。
   - `videosar/mwp.py`：独立封装微波光子（MWP）链路退化模块（激光器相噪、MZM 调制非线性、光电探测散粒噪声与热噪声），默认关闭（纯 Ka 波段 Video-SAR），需要做微波光子消融时通过 `--mwp-enabled` 或 `MWPConfig(enabled=True)` 显式开启。

4. **工程卫生与可复现实验协议**
   - 清理指向未提交路径（`runs/snl-domain-ablation/`）的 404 死链；所有实验统一输出 `config.json`、`metrics.json`（含 `git_commit`、`seed`、`timestamp_utc` 并严格分区记录 `real_blind_eval` 与 `synthetic_ablation`）、`summary.csv` 与 `froc_curve.csv`。

---

## 二、分区验收规范（严格分离）

### 1. 实测盲测基准（主门槛）
| 数据集 | 切分方式 | 评测输出分区 | 参考来源 |
| :--- | :--- | :--- | :--- |
| **VSMB** | 按序列 ID 严格互斥切分（70/15/15） | `metrics.json["real_blind_eval"]` | [GitHub](https://github.com/softwarePupil/VSMB) / [arXiv:2506.12105](https://arxiv.org/html/2506.12105) |
| **SNL Eubank Gate** | 时间块切分（1–600 训练，+16 帧隔离带，617–900 测试） | `metrics.json["real_blind_eval"]` | [Sandia Footage](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4) |

### 2. 物理仿真与微波光子消融支线（独立记录）
- 默认模式：`backend="real_bg_injection"`, `resolution>=256`, `mwp.enabled=False`
- 微波光子消融模式：`mwp.enabled=True`（记录 `mwp_mean_evm_rms` 与检测/跟踪退化对比）
- 射线追踪回波模式：`backend="radarsimpy"`（需单独安装 [RadarSimPy](https://github.com/radarsimx/radarsimpy)）

---

## 三、快速开始

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 运行全套单元测试（验证防泄露、空场景无NaN、阴影注入、MWP退化与MOT评测）
python3 -m unittest discover -s tests -v

# 3. 运行端到端冒烟评测（生成标准化 metrics.json / summary.csv / froc_curve.csv）
python3 scripts/run_benchmark.py --mode smoke --output-dir outputs/smoke_baseline

# 4. 运行微波光子链路退化消融并导出 MOT 格式仿真序列
python3 scripts/run_benchmark.py --mode simulate --mwp-enabled --resolution 256 --output-dir outputs/mwp_ablation

# 5. 加载本地已下载的 VSMB 或 SNL 实测集
python3 scripts/run_benchmark.py --mode eval-vsmb --data-root /path/to/VSMB
python3 scripts/run_benchmark.py --mode eval-snl --data-root /path/to/SNL
```

---

## 四、历史模块导航

`videosar/` 中仍保留一代研究 CLI 脚本（`stress`、`aperture-study`、`center-study`、`small-shadow`、`snl_causal` 等），它们由 `python -m videosar` 调用，用于复现既有 `deliverables/` 结果，**不属于 v2.0 评测主链路**。这些脚本相互 import，且与部分交付目录耦合，当前不做迁移。

哪些文件属于哪一代、依赖如何、为何暂不归档，见 [docs/legacy-module-index.md](docs/legacy-module-index.md)。新实验请使用上文的 `scripts/run_benchmark.py` 入口。

---

## 五、交接说明

项目当前分支、工作区、数据资产限制、红线与待办清单见 [docs/HANDOFF.md](docs/HANDOFF.md)。接手前请先阅读，尤其是「数据资产现状」中 SNL 真值对齐与 VSMB 数据缺失两节。
