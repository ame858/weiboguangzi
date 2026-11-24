# Deliverables Protocol (v2.0)

本目录用于归档可复现的实验交付物。旧版指向未提交路径（如 `runs/snl-domain-ablation/`）的死链报告已全部清理。

每次通过 `videosar.experiment.ExperimentRunner` 或 `scripts/run_benchmark.py` 运行后，都会在指定输出目录下生成自包含的四件套：

- `config.json`：完整实验配置快照（含分辨率、雷达参数、MWP 开关与切分参数）
- `metrics.json`：附带 `git_commit`、`seed`、`timestamp_utc`，并严格拆分为 `real_blind_eval` 与 `synthetic_ablation` 两大分区
- `summary.csv`：跨切分核心指标汇总表（`map_50`, `map_30`, `fa_per_frame_50`, `recall_at_1_fa`, `mota`, `idf1`, `hota`, `id_switches`）
- `froc_curve.csv`：完整 FROC 曲线采样点（`fa_per_frame`, `recall`, `score`）
