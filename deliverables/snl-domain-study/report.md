# SNL 仿真尺寸与遮挡衰减配对试验

四组时序 U‑Net 在新锁定的 SNL 三段视频、24 帧、169 个逐帧标注框上均命中 **0/169**。只减弱遮挡衰减提高了已知位置的阴影排序，但四组均未达到全图检测目标，联合配置未通过采用条件。[SNL 原视频](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4) · [公开框标注](https://github.com/Jiangxb8/Video-SAR/tree/28a6ee2cff7913d817816c01589e46c862927df0)

## 数据和校准

复用同一批 36 个有目标场景和 12 个无目标场景。四组共享背景、散射场、轨迹、孔径、接收噪声和裁剪位置。每组使用相同的 69 个正窗口和 69 个负窗口，窗口为 48×48。各组均使用项目时序 U‑Net，训练 14 轮，随机种子为 1729 和 2027。使用原交叉实验的训练集标准化统计量。

从仿真训练场景中固定 12 个做校准，只用 SNL 的旧训练段定目标值，框面积中位数 112.38 像素²，归一化对数灰度的框内外对比中位数 0.112。选出的长、宽、高倍率为 1.85，线性功率遮挡衰减保留 45%。仿真训练帧使用施加合成平移前的坐标，确保图像和标注对齐。对比度组沿用同尺寸的原衰减框。C 组 69 个正窗口中有 7 个失去原可见标签的全部重叠，D 组 69 个窗口均仍有重叠。[校准网格](../../runs/snl-domain-ablation/calibration.csv) · [训练分布](../../runs/snl-domain-ablation/training-distribution.csv) · [配对窗口](../../runs/snl-domain-ablation/paired-windows.json)

## 锁定验证与盲测

用 SNL 原有 12 帧验证集在 0.1–0.9 网格选择阈值。每帧取平均前景分数最高的至多 20 个连通域外接框，以 IoU ≥ 0.50 一对一匹配。验证网格内各组均无匹配。按约定的并列规则，八个模型均冻结阈值 0.9。盲测标注编号为 332–374、560–602、830–872，每隔 6 帧取一帧。24 个标注的 Git 对象哈希均核对一致，原视频映射帧与标注参考图的缩略图相关系数为 0.9919–0.9980。[验证网格](../../runs/snl-domain-ablation/validation.csv) · [冻结记录](../../runs/snl-domain-ablation/frozen.json) · [盲测对齐](../../runs/snl-domain-ablation/blind-alignment.json)

| 训练组 | 框面积中位数（像素²） | 框内外对比中位数 | 命中（每个种子） | 误报（两个种子） | 已知位置排序胜率（两种子平均） |
|---|---:|---:|---:|---:|---:|
| A 原样 | 32 | 0.285 | 0/169 | 4、0 | 44.08% |
| B 放大尺寸 | 112 | 0.323 | 0/169 | 0、9 | 48.22% |
| C 减弱对比度 | 32 | 0.112 | 0/169 | 0、0 | 55.33% |
| D 两项联合 | 112 | 0.129 | 0/169 | 0、0 | 43.20% |

在已知位置排序中，每个真实框配一个同尺寸、附近且与其他标注框无重叠的背景框，比较框内的平均前景分数。C 相对 A 在两个种子上分别改善 **14.79** 和 **7.69 个百分点**，均值改善 **11.24 个百分点**。B 的两种子增益方向相反。D 相对 A 的平均差为 **−0.89 个百分点**。表中的排序需要已知位置，仅用于辨别输入响应。各组在阈值冻结条件下的全图召回与 F1 都是 0%。没有候选时查准率未定义。结果适用于当前 SNL 单一场景、训练预算、框监督和候选提取规则。[分段指标](../../runs/snl-domain-ablation/blind-results.csv) · [逐帧数据](../../runs/snl-domain-ablation/blind-per-frame.csv) · [配对差与交互](../../runs/snl-domain-ablation/paired-effects.csv) · [采用检查](../../runs/snl-domain-ablation/blind-status.json)

## 复现

在仓库根目录运行。实验读取此前保存的仿真场景和 SNL 数据，24 个锁定测试标注位于 `runs/snl-domain-ablation/labels/`。FFmpeg 可用绝对路径指定。已有冻结权重时可加 `--score-only` 复算盲测。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe deliverables\snl-domain-study\prepare.py
.\.venv\Scripts\python.exe deliverables\snl-domain-study\run.py --ffmpeg 'C:\path\to\ffmpeg.exe'
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

配对衰减输出与独立重算的回波仿真一致，原衰减组逐像素相同。26 项自动测试通过，逐帧计数、阈值选择、模型哈希和汇总指标已复核。
