# SNL Sim-to-Real 对照实验报告

> 生成时间：2026-11-25
> 代码分支：`feat/snl-sim2real-contrast`
> 相关脚本：`scripts/run_snl_sim2real.py`、`scripts/build_proposal_dataset.py`

## 一、实验协议（严格时间块切分，无泄露）

SNL Eubank Gate 的可用标注只覆盖 **frame 150–527**，故放弃标准 617–900 协议，改用标注区内的严格时间块切分：

| 区段 | 帧范围 | 帧数 | 说明 |
| :--- | :--- | :--- | :--- |
| train | 150–399 | 250 | 真实 GT 2159 框；真实底图**只从此段**提取 |
| guard gap | 400–415 | 16 | 跳过，防滑窗泄露 |
| test | 416–527 | 112 | 真实 GT 910 框，唯一评测区 |

- 真实底图提取窗口：`--snl-min-train-frame 150 --snl-max-train-frame 399`，manifest 记录 `snl_train_frame_window=[150,399]`
- 训练集生成：`build_proposal_dataset.py` 全量 120 序列 / 3000 帧，真实底图仅注入 id_train 的 39 组，其余分区为 0
- `train_norm_stats` 仅由 id_train 统计
- git_commit / seed 随产物记录

## 二、对照设计（三组，同一冻结测试集）

检测器：零训练因果时序中值背景差分 `detect_moving_shadows_baseline`。各组只在该组自己的训练数据上标定 `z_thresh`，再统一在 416–527 上评测。

| 组 | 训练数据 | 含义 |
| :--- | :--- | :--- |
| `synth_only` | 纯物理仿真序列（无真实底图） | 纯 sim→real 迁移 |
| `snl_train` | 真实 SNL 训练帧 150–399 | 仅 SNL 训练 |
| `snl_plus_aug` | 真实 SNL 训练帧 + 真实底图注入仿真扩增 | 申报书期望的增强组 |

## 三、结果：三组 mAP@0.5 全为 0，**无有效增益可报告**

```
test frames [416, 527] | num_gt=910
  synth_only:   z=1.2  mAP50=0.0000  FA/frame=142.79  recall=0.0055
  snl_train:    z=1.2  mAP50=0.0000  FA/frame=142.79  recall=0.0055
  snl_plus_aug: z=1.2  mAP50=0.0000  FA/frame=142.79  recall=0.0055
gain snl-synth=+0.0000 | aug-snl=+0.0000
```

补充：在 IoU=0.3（小目标阈值）下重评，最高仅 recall 0.022、AP@0.3 ≈ 0.0001，同样无效。

## 四、根因诊断（**结论：检测器本身在真实 SNL 上失效，不是"没增益"**）

1. **GT 框极小**：车辆目标 6×11 ~ 15×24 像素，面积中位 139 px。
2. **检测框与 GT 空间不吻合**：实测单帧最大 IoU 仅 0.23–0.29，达不到 0.5（甚至 0.3）匹配阈值。检测出的 6393 个框全是散斑连通域，没有一个对应真目标。
3. **帧间信号太弱**：相邻帧平均绝对差仅 1.2 灰度级（≈0.47%），目标阴影只占几个像素，中值背景差分在真实 SAR 相干斑上信噪比不足。
4. **排除配准问题**：相位相关测得帧间全局位移 < 1 px，不是机载运动补偿缺失。

因此三组数值完全一致并非"增强无效"，而是**底层检测器在该数据工况下输出全是噪声**——任何阈值标定都无法产生有意义的 mAP。**此结果不可作为 Sim-to-Real 增益证据，也不应写入申报书/论文。**

## 五、产物

- `sim2real_report.json`：完整对照记录（含 git_commit、seed、帧范围、三组指标、每档 z 的调参曲线）
- `sim2real_contrast.csv`：对照表
- `config.json` / `metrics.json` / `summary.csv` / `froc_curve.csv`：`ExperimentRunner` 标准分区产物
  - 注：`metrics.json` 与 `froc_curve.csv` 中的 FROC 曲线已下采样（检测器失效导致每 split 约 16000 个噪声框、每框一个点，原始 9.7MB）。标量指标（mAP/recall/FA）未改动，完整曲线可由脚本重跑复现。


## 六、下一步建议

1. **换检测器**：中值背景差分不适合该数据。需要能利用目标形态/对比度的检测器（如局部对比度、深度检测器），才能建立有意义的 Sim-to-Real 对照。
2. **或改评测口径**：若坚持传统基线，应先做帧间配准 + 累积检测，并重新标定 min_area/max_area 以匹配 6–24 px 的小目标。
3. **补齐标注**：若要恢复标准 617–900 协议，需补充该区段的 GT。
4. 在检测器能产出非零 mAP 之前，**不要**把 Sim-to-Real 对照表写入交付材料。
