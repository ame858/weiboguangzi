# SNL Eubank 因果检测实验

本实验对 SNL Eubank Gate 的原生 660×720 雷达区域预测车辆阴影框。每帧输入限于当前帧和历史画面。在 IoU ≥ 0.50 时按逐帧一对一匹配统计查准率、召回率和 F1。训练与开发帧取自同一段视频，正式成绩用固定时间段的封存标注验收。

## 视频文件推理

`python -m videosar detect-video` 使用随目录保存的七帧模型配置与权重。配置记录检查点 SHA‑256 和开发成绩。运行时提供视频文件即可，当前支持指定矩形雷达区和源帧范围。处理开始时从选中首帧建立历史，输出可覆盖完整视频或所选前缀。需要 PyTorch、Torchvision、SciPy 与支持 `libx264` 的 FFmpeg。FFmpeg 可从 PATH 获取，也可用 `--ffmpeg` 指定。也可在项目环境中安装 `imageio-ffmpeg` 使用随包二进制。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m videosar detect-video --input runs\snl-eubank-check\eubank-snl.mp4 --crop 310 0 660 720 --start-frame 150 --max-frames 378 --output-dir runs\snl-video-demo
```

该命令处理 SNL 雷达画面从零基源帧 150 起的 378 帧。预测 CSV 同时记录输出帧号和源帧号，框的坐标位于裁剪后的雷达网格。叠框视频按源流的标称帧率编码为无音频 MP4。所附权重用于研究演示，当前可量化的精度来自 SNL 开发复核。[模型配置](models/stacked-center.json)

378 帧复跑生成同帧数的叠框 MP4，重读坐标 CSV 得到原开发段的 102 个命中、176 个误报和 347 个漏检。优化七帧历史读取后的同机 CPU 复跑用时约 215 秒，帧清单、预测 CSV 和 MP4 均与优化前逐字节一致。带正式验收标签的评测仍按后文的封存协议走。[378 帧叠框](../../runs/snl-video-prefix-v2/overlay.mp4) · [复算与文件校验](../../runs/snl-video-prefix-v2/development-replay.json)

另外用原始尺寸 256×192、时长 192 秒的仿真循环视频测试整段读写。输入和输出均为 960 帧，帧清单与坐标表的计数一致。这一轮用于验证长文件处理。[960 帧文件校验](../../runs/snl-video-long-v2/file-validation.json)

## 运行

从仓库根目录运行。现有项目环境支持 CPU。CUDA 可用时脚本自动用 GPU。输入复用 `runs/snl-eubank-check/` 中的官方 MP4、Git 标注清单和首段已校验标注。Faster R‑CNN 初始权重需放入 `runs/snl-causal-study/`，文件名包含其 SHA‑256 前缀。

```powershell
$py = '.\.venv\Scripts\python.exe'
$f = 'C:\path\to\ffmpeg.exe'
& $py deliverables\snl-causal-study\acquire.py development
& $py deliverables\snl-causal-study\experiment.py verify --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py baselines --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py train --method gray --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py train --method temporal --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py develop --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py iterate --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py center --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py causality --ffmpeg $f
```

校准段与开发复核段都达到查准率、召回率双 80% 后，从同一输出目录检验前缀因果性并冻结设置。冻结记录保存权重校验值、视频校验值和推理源码校验值。取得封存标注后，评分流程先生成检测框，再读取标注并匹配。首套失败后才可以冻结新的合格配置使用备用段。

```powershell
& $py deliverables\snl-causal-study\experiment.py causality --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py freeze --ffmpeg $f
& $py deliverables\snl-causal-study\acquire.py first
& $py deliverables\snl-causal-study\experiment.py blind --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py render --ffmpeg $f
```

开发阶段通过三个检查点比较单帧灰度和因果时序 Faster R‑CNN。因果输入包括当前灰度、6 帧暗化和历史背景残差。候选不足时尝试三倍输入与更小的锚框。中心检测器用同一套实测框监督，直接学习多个中心、亚像素偏移和框尺寸。视频实际可解码的 SAR 区域有 899 帧。

## 扩充早期训练段

前向划分增加了较晚的实测训练帧，拟合区间为编号 2–296，校准区间为 310–323，开发复核仍为 332–379。两套正式验收区间保持封存。新划分使用独立输出目录。将同样的 `--profile` 和 `--output` 传给后续验证、因果性检查和盲测命令。

```powershell
$run = 'runs\snl-causal-study\prospective'
& $py deliverables\snl-causal-study\acquire.py development --profile prospective --output $run
& $py deliverables\snl-causal-study\experiment.py center-domain --profile prospective --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\preview.py --output $run --ffmpeg $f --fps 8
```

可独立比较基于因果时间片生成的运动阴影、局部标准化、历史帧配准和邻域框回归。相应训练命令依次为 `center-motion`、`center-normalized`、`center-registered` 和 `center-iou`。后缀一致性检查覆盖配准分支与新输入配置。

`center-stacked` 使用当前帧及滞后 1、3、6、12、24、30 帧。`center-stacked-motion` 在相同的滞后位置模拟运动阴影，背景和物理参数样本均来自早期训练。`center-background` 在同一输入上增加因果背景残差。背景网络只用早期训练段拟合，标签框及外扩 3 像素区域不计入背景误差。`center-spatial-views` 在校准段选定的检查点上比较镜像图融合和图块。`center-space-depth` 将常规步长卷积替换为像素重排后再卷积，比较细节保留对小目标框的影响。各命令沿用同一标注和 IoU 门槛。[DSENet](https://www.mdpi.com/2072-4292/18/10/1623)

`center-occupancy` 利用已经训练的框内占用分支重新计算置信度。`center-shared-temporal` 对各滞后帧共享空间编码，再对当前与历史特征做差。`center-line-context` 增加原生图像横纵方向的条带响应，检验拖尾线索能否帮助区分静止暗区与运动阴影。线特征对照参考 VSMT 对拖尾的建模思路，保留现有中心检测器。[VSMT](https://arxiv.org/html/2506.12105v1#S4.SS2)

`center-weak-shadows` 使用早期真实标注，在各滞后帧内逐框减弱阴影对比度。对同一时序图块使用相同强度系数，并保留基础对照的抽样顺序、初始化与各项训练设置。

`center-weak-sampling` 直接增加低对比度和低空间频次的真实训练目标被裁入训练窗口的概率。原始像素和坐标沿用同一来源，采样权重从早期训练集计算。

`center-box-projection` 增加框投影和相邻像素一致性作为前景辅助监督，检测框标签与匹配口径沿用原设置。[BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)

`center-local-gain` 训练时对整块背景和阴影施加一致的局部对比度变化，同一时序图块的七帧共享幅度。该分支检验对比度适应能否从推理增强变成模型自身的能力。

`center-contrast-views` 融合原图与增强局部对比度后的预测。`center-stacked-normalized` 对七个时刻分别按局部背景和方差标准化。`center-quality-focal` 用预测框与标注框的当前 IoU 监督中心分数，使排序分数反映定位质量。[Generalized Focal Loss](https://arxiv.org/abs/2006.04388)

`track-support` 复用已有校准和开发预测，比较需要两帧或三帧历史强框支持的弱候选，校准时同时保留只用高阈值的选项。开发预览直接读取评分时使用的候选，输出左右对照视频及逐框 CSV。预览脚本会核对输出的命中、误报和漏检计数与决策记录一致。

`center-stitch` 在重叠图块上计算中心概率与框几何，在原图网格融合后统一解码。图块边缘使用较小融合权重，像素坐标回到原始 SAR 裁剪区域。

`center-stacked-long` 保持七帧输入，把历史间隔改为 1、6、15、30、60、90 帧，用于检验更早的背景观测是否帮助分离低速车辆阴影。

`audit_spatial_alignment.py` 通过局部灰度相关性核验公开标注图和原视频的空间映射。`audit_detection.py` 统计实际输出、放宽分数后的候选以及用真值框宽高进行反事实替换的覆盖率。两项诊断使用开发时段，正式指标采用原始框与固定匹配口径。

`snl_low_rank.py` 比较因果低秩重构、带稀疏异常抑制的低秩重构和局部对比度加权版本，阈值和面积门槛从校准段选择。

`snl_forward_ranking.py` 在编号 242–296 的早期帧上训练候选重排器。生成这组候选的检测器只见过编号 2–218。之后将重排器用于扩充训练后的候选，使用 310–323 校准，并保持 332–379 的开发复核区间。

`snl_data_budget.py` 用同一套校准及开发帧重算 217 帧和 295 帧训练的七帧模型。`snl_box_calibration.py` 在校准段拟合框宽高缩放，`--image-center` 改为对比由图像局部暗度与历史背景残差支持的小幅平移。

`audit_paper_examples.py` 比较 DSENet 图 2 的示例框与相同早期图像的社区框，记录近邻帧对齐及中心和 IoU 分别匹配的情况。此结果用于说明标注口径，论文的完整训练或评测标注仍需另行取得。[DSENet 图 2](https://www.mdpi.com/2072-4292/18/10/1623)

`jiang-faster-rcnn` 使用标注仓库的全图 Faster R-CNN ResNet-50-FPN 架构、默认锚框、COCO 初始化和单前景类别。复跑时以公开视频解码帧为输入，沿用本实验的时间划分和操作点搜索，从 COCO 检查点起训。上游脚本使用随机切分并依赖其本地续训权重。`full-small-anchors` 保留全图架构，将先验锚框调到更接近小阴影尺度和长宽比的范围。[标注仓库训练代码](https://github.com/Jiangxb8/Video-SAR/blob/28a6ee2cff7913d817816c01589e46c862927df0/deep_learning.py)

`yolo11` 用 Ultralytics 8.3.200 和公开的 YOLO11s COCO 权重做全图单帧对照。安装该版本及其依赖后，将官方 `yolo11s.pt` 放入 `runs/snl-causal-study/`。`yolo11-scratch` 从同一网络的随机初始化起训，用于和 DSENet 论文的训练条件进行对照。训练数据导出只包含早期训练帧和校准帧。脚本将权重和参数记录到同一实验目录，并沿用一对一 IoU ≥ 0.50 匹配。[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) · [固定版本官方权重](https://huggingface.co/Ultralytics/YOLO11/blob/a01aaa06caeff788b052e193acb76b3f21571b3a/yolo11s.pt)

`yolo11-temporal` 保留同一预训练架构，三个输入通道分别为当前灰度、滞后 6 帧的暗化量与历史背景残差。训练时关闭色相和饱和度扰动，以保留三个通道的物理含义。

`yolo11-p2` 在相同单帧输入上将步长 4 的细节特征接入检测头，沿用兼容的官方 YOLO11s 预训练层，用于对照小阴影的空间采样瓶颈。[上游 P2 配置](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/cfg/models/v8/yolov8-p2.yaml)

`snl_detector_agreement.py` 用各自校准阈值归一化中心检测器与 YOLO11 的原始分数。候选在两个模型间有框重叠时，可以按校准选定的比例增信，并保留单来源候选供评测。

`snl_proposal_audit.py` 汇总多模型候选，并将各候选的尺寸保持一致、在画面内随机重新放置中心作为对照。分析分别计算中心距离 5 像素和框 IoU ≥ 0.50 的一对一覆盖上限。

主要记录为 `development-status.json`、逐模型开发记录、`causality-check.json` 和正式盲测结果。冻结前的开发记录用于迭代，正式指标来自独立解封后的坐标匹配。

[SNL 官方视频](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4) · [公开标注](https://github.com/Jiangxb8/Video-SAR/tree/28a6ee2cff7913d817816c01589e46c862927df0) · [Torchvision 预训练权重](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)

## SIVED 实测 SAR 预训练对照

对比两个 YOLO11s 初始化。对照组从官方 COCO 特征层开始，迁移组先用 SIVED 的静态车辆图像继续预训练，然后转移特征层。两组使用同种子重新初始化 SNL 检测头，复用同一 SNL 灰度输入、训练预算和框匹配规则。分别用校准段选择检查点及操作点，再在 48 帧开发复核段比较。使用同一批帧的计数计算 F1 差异，按六帧时段分块重采样。正式验收遵循原封存流程。[SIVED 数据集和原始归档](https://github.com/CAESAR-Radi/SIVED/tree/97616f4a6497721258652926762d8234f073c14d) · [SIVED 论文](https://doi.org/10.3390/rs15112825)

将作者的 `SIVED.zip` 放在 `runs/snl-causal-study/external-sar/`。脚本核对归档的 Git LFS SHA‑256，将作者训练与验证标签的旋转框转换为轴对齐外接框。训练集为 837 张图、9,561 个框，验证集为 104 张图、1,222 个框。两集合计覆盖 Ka、Ku、X 三波段。沿用前文 `$py`、`$run`、`$f`，其中 `$py` 指向已装 Ultralytics 8.3.200 的环境。

```powershell
& $py deliverables\snl-causal-study\snl_sived.py prepare
& $py deliverables\snl-causal-study\snl_transfer_compare.py --audit-data --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\snl_sived.py fit
& $py deliverables\snl-causal-study\experiment.py yolo11-coco-transfer --profile prospective --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\experiment.py yolo11-sived --profile prospective --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\snl_transfer_compare.py --output $run
```

训练分布审计以原生图像的四像素外环灰度中位数减去框内中位数。SIVED 的 9,561 个训练框中有 150 个比外环暗，面积中位数为 1,566 像素²。SNL 早期的 2,554 个训练框中有 2,502 个比外环暗，面积中位数为 110 像素²。两组原生像素尺度只用于描述检测器输入，物理尺寸需要各自的地面采样尺度。[训练外观审计](../../runs/snl-causal-study/prospective/sived-snl-training-appearance.json)

两组均使用种子 1729 和 295 帧 SNL 图像训练，按同一早停规则分别运行 46 和 38 轮。SIVED 源任务预训练 25 轮，选定检查点的 SIVED 验证 AP50 为 96.69%。下表使用 SNL 标注编号 332–379 的 48 帧、449 个框，阈值及检查点由编号 310–323 的 14 帧校准段选择。[SIVED 预训练记录](../../runs/snl-causal-study/external-sar/sived-fit.json) · [SNL 配对评分和检查点](../../runs/snl-causal-study/prospective/sived-transfer-comparison-seed1729.json)

| SNL 开发复核方法 | 命中 | 误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11s，COCO 特征层 | 63 | 228 | 21.65% | 14.03% | 17.03% |
| YOLO11s，追加 SIVED 预训练 | 68 | 166 | 29.06% | 15.14% | 19.91% |
| 七帧中心检测与局部对比度融合 | 102 | 176 | 36.69% | 22.72% | 28.06% |

追加 SIVED 预训练使 F1 提高 2.89 个百分点，六帧分块重采样的 95% 区间为 −0.99 至 +6.68 个百分点。单种子、单场景的结果仍有较大不确定性。用现有候选一致性方法合并两模型后，中心检测器与 COCO、SIVED 分支的 F1 分别为 22.79% 和 21.14%。汇总中心检测器和 SIVED 分支全部 16,576 个导出候选框，忽略操作阈值并直接寻找最优匹配，最多命中 232/449，覆盖 51.67%。这给出了该候选池在当前 IoU 门槛下的重排上限。[COCO 融合记录](../../runs/snl-causal-study/prospective/center-coco-transfer-agreement.json) · [SIVED 融合记录](../../runs/snl-causal-study/prospective/center-sived-transfer-agreement.json)

当前保留七帧中心检测器与局部对比度融合。该配置已通过前缀因果检查，开发复核仍低于正式门槛，两套盲测均处于封存状态。[开发段叠框视频](../../runs/snl-causal-study/prospective/development-preview.mp4) · [因果检查](../../runs/snl-causal-study/prospective/causality-check.json)

### 源图像对比度方向对照

在 SIVED 原始预训练对照上，仅将 837 张训练图和 104 张验证图按 `255 − RGB` 反相并无损保存，尺寸与框标注沿用原数据。预训练、SNL 微调和阈值选择使用原设置。反相样本已逐张复算核对像素，标注与原样逐字节一致。[数据校验](../../runs/snl-causal-study/external-sar/sived-inverted-pair-check.json)

训练框比局部背景暗的比例由 1.57% 增至 98.12%，接近 SNL 训练框的 97.96%。反相组在 SNL 开发复核命中 83 框、误报 297 框，查准率 21.84%、召回率 18.49%、F1 20.02%。原 SIVED 组对应为 68、166、29.06%、15.14% 和 19.91%。F1 差为 +0.11 个百分点，六帧分块重采样的 95% 区间为 −2.87 至 +3.10 个百分点。反相组与七帧模型的全部导出候选合计最多命中 242/449，原 SIVED 组与七帧模型合计为 232/449。[外观统计](../../runs/snl-causal-study/prospective/sived-inverted-snl-training-appearance.json) · [配对评分](../../runs/snl-causal-study/prospective/sived-inverted-transfer-comparison-seed1729.json)

按同一校准段选融合配置后，反相组与七帧模型合并输出命中 88 框、误报 281 框，F1 21.52%。交付继续使用七帧模型的单路预测。[融合记录](../../runs/snl-causal-study/prospective/center-sived-inverted-transfer-agreement.json)

```powershell
& $py deliverables\snl-causal-study\snl_sived.py fit --invert
& $py deliverables\snl-causal-study\experiment.py yolo11-sived-inverted --profile prospective --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\snl_transfer_compare.py --invert --output $run
```

### 源预训练目标尺度对照

按 YOLO 等比缩放、补齐画布的规则估算网络输入中的目标框面积，原始 SIVED 用 512 像素输入时中位数约为 1,566 像素²，SNL 用 768 像素输入时约为 125 像素²。另将原始 SIVED 以 160 像素预训练，预计中位数约为 153 像素²。保持原灰度极性，SNL 两组均沿用 768 像素输入和同一微调设置。

```powershell
& $py deliverables\snl-causal-study\snl_sived.py fit --image-size 160
& $py deliverables\snl-causal-study\experiment.py yolo11-sived-160 --profile prospective --output $run --ffmpeg $f
& $py deliverables\snl-causal-study\snl_transfer_compare.py --small-source --output $run
```

### 同一时序输入下的预训练对照

在原始、反相、160 像素三种 SIVED 权重中，按单帧 SNL 校准段的 F1、召回率和查准率选择时序试验的初始化。对照组从 COCO 特征层起步。两组均重新初始化检测头，训练输入统一为当前灰度、滞后 6 帧暗化量和历史背景残差，使用同一批 SNL 帧与训练设置。选定阈值后分别报告开发复核结果。

`yolo11-coco-transfer-temporal` 运行对照。`yolo11-sived-temporal`、`yolo11-sived-inverted-temporal` 和 `yolo11-sived-160-temporal` 分别使用对应的 SIVED 初始化。评分时向 `snl_transfer_compare.py` 传 `--temporal`，反相或 160 像素组再分别加 `--invert` 或 `--small-source`。

## 开源 ByteTrack 对照

固定七帧中心检测器及局部对比度融合的候选，给两个时间段各保留前六帧供跟踪器初始化。比较默认 ByteTrack、校准段选参的 ByteTrack，以及保留原强框并只补充有轨迹支持的弱框。使用 Ultralytics 8.3.200 中的上游实现和 `lap==0.5.12`。跟踪器在开发段的前缀输出逐项相同。[ByteTrack 论文](https://arxiv.org/abs/2110.06864) · [固定版本实现](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/trackers/byte_tracker.py)

| 同一批候选的输出方式 | 命中 | 误报 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: |
| 原检测器 | 102 | 176 | 22.72% | 28.06% |
| 默认 ByteTrack | 27 | 37 | 6.01% | 10.53% |
| 校准后 ByteTrack | 75 | 154 | 16.70% | 22.12% |
| 保留强框并补充弱框 | 102 | 203 | 22.72% | 27.06% |

开发复核有 48 帧和 449 个真值框。补弱框方案额外产生 27 个误报，命中仍为 102 个。保留原检测器作为本组对照的首选。[逐帧指标与分块区间](../../runs/snl-causal-study/prospective/bytetrack-development.json)

```powershell
& $py deliverables\snl-causal-study\snl_bytetrack.py --output $run --ffmpeg $f
```

## 已发表方法的验收口径

DSENet 的主结果表列示查准率 92.40%、召回率 75.50%、F1 83.10%。同一表的 538 个命中和 690 个真值可复算出召回率 77.97%，与正文另一处一致。MambaShadowDet 在其 1,694 个真值上列示查准率 88.90%、召回率 73.26%、F1 80.32%。论文各自使用的真值与划分不同。这两篇论文的列示召回均低于本项目 80% 门槛，同一 SNL 开发协议的开源复跑以本项目逐帧统计为准。[DSENet 表 2、表 3](https://www.mdpi.com/2072-4292/18/10/1623) · [MambaShadowDet 表 1](https://www.mdpi.com/2072-4292/17/2/214)

## 标注与漏检复核

已核验原视频、社区 LabelMe 框、四篇论文和两个候选公开资料库。现成可复跑的同域对照为 Jiangxb8 的 Faster R-CNN，按当前开发协议得到 F1 19.43%。完整的论文标注、精确划分和官方权重仍待取得。VSMB 发布入口未能提供可核验归档。[论文材料与开源对照](research-materials.md)

在三个公开时间段各抽首、中、末一帧，共覆盖 78 个框。九帧映射均在邻近五帧的相关比较中排名第一，相关系数 0.9920–0.9979。9 帧的高相关局部图块位移中位数为 0.67–0.83 个社区参考图像像素。已有的训练段仿射敏感性检查将开发匹配数从 102 降至 94。本轮沿用原映射和社区标注协议。九帧抽查验证了帧与坐标，大范围的语义边界一致性需要独立标注才能评估。[九帧叠框](../../runs/snl-causal-study/prospective/annotation-audit-nine.png) · [坐标核验](../../runs/snl-causal-study/prospective/annotation-audit-nine.json) · [仿射敏感性](../../runs/snl-causal-study/prospective/detection-error-audit-radar-center-stacked-epoch36-contrast1.35-support3.json)

| 开发段逐框结果 | 数量 | 标注框宽×高中位数 |
| --- | ---: | ---: |
| 已检测 | 102 | 8.48×17.69 px |
| 有合格原始候选，最终未输出 | 42 | 8.48×13.07 px |
| 原始候选需调整位置或尺寸 | 126 | 8.48×16.92 px |
| 有限几何调整后仍缺可匹配候选 | 179 | 8.48×14.61 px |

逐帧最大匹配保留前一阶段已覆盖的真值。几何调整枚举每轴 −2、0、+2 px 平移及 0.8、1.0、1.2 倍缩放，始终约束每个候选只匹配一个真值。使用真值进行匹配的候选上限从 144/449 增至 270/449，低于 80% 召回所需的 360/449。单独继续扫描这组候选的阈值无法达到目标。[逐框 CSV](../../runs/snl-causal-study/prospective/development-miss-attribution.csv) · [分类明细](../../runs/snl-causal-study/prospective/development-miss-attribution.json) · [代表帧](../../runs/snl-causal-study/prospective/development-miss-examples.png)

```powershell
& $py deliverables\snl-causal-study\snl_research_audit.py --output $run --ffmpeg $f
```

## 仿真预训练配对对照

使用申报书仿真数据的 72 段训练序列预训练 8 轮，其中图像域 48 段、回波成像 24 段，共 1,440 帧。在 192×192 图块内保留仿真的原始像素尺度，采用边缘填充、每段 2–98 百分位灰度映射，并按已记录的扰动位移同步移动掩膜。每段只有 20 帧，因此滞后 24 和 30 帧的通道均按七帧输入的短历史规则回填当前帧。

预训练集共有 1,200 个非空掩膜外接框，宽高中位数为 6×13 px。SNL 训练集 2,554 个框的中位数为 7.82×14.83 px。以框外 4 px 环带灰度中位数减去框内中位数，仿真与 SNL 的对比度中位数分别为 0.2451 和 0.0353，均在 0–1 灰度尺度计算。该差距描述本次预训练数据与社区框样本的视觉差异。

两组 SNL 阶段使用相同的七帧中心检测器、295 帧标注、种子 1729、48 轮优化预算和每 12 轮一次的检查点。两组在同一 CUDA 环境重新训练，以 14 帧校准段选择检查点与检测操作点。仿真源训练使用 CPU，迁移组加载其已校验的检查点。开发段评测沿用 48 帧、449 个目标和逐帧一对一 IoU ≥ 0.50。

| SNL 训练方式 | 校准 F1 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 | 全部候选覆盖上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 直接训练 | 23.17% | 61 | 67 | 47.66% | 13.59% | 21.14% | 131/449 |
| 仿真预训练后微调 | 26.67% | 62 | 96 | 39.24% | 13.81% | 20.43% | 117/449 |

仿真组在开发段多命中 1 个、多误报 29 个，F1 变化为 **−0.72 个百分点**。六帧分块配对重采样的 95% 区间为 **[−5.18，+3.19] 个百分点**。本次数据与适配方式没有显示稳定的开发收益，全部候选覆盖也低于 80% 召回所需的 360 个。继续使用查准率 36.69%、召回率 22.72%、F1 28.06% 的七帧交付模型，两套正式时段继续封存。[配对结果和检查点哈希](../../runs/snl-causal-study/prospective/simulation-pretraining-comparison.json)

```powershell
$studyPy = '.\runs\snl-causal-study\sived-env312\Scripts\python.exe'
& $studyPy deliverables\snl-causal-study\snl_simulation_transfer.py --output $run --dataset runs\proposal-final-data --ffmpeg $f
```

本机完成 SNL 阶段所用 PyTorch 为 2.9.1+cu128。重放同一次实验可复用 `$run\simulation-pretraining` 内的源检查点和两组 SNL 检查点。

## 因果时序与小框候选对照

在相同 295 帧训练段和 14 帧校准段上，比较卷积循环时序编码、当前帧与跨帧差分的双路编码，以及 Torchvision RetinaNet。两种时序变体各训练 48 轮。RetinaNet 用配准后的因果三通道，在三倍图块上用 P2 的多尺度小锚框产生候选，训练 12 轮。表中每组的检查点和操作点由校准段选定，开发段共有 449 个标注框。卷积循环和双路编码分别参考红外小目标工作中的时序建模思路实现。[SSTNet](https://github.com/UESTC-nnLab/SSTNet) · [STMENet](https://github.com/UESTC-nnLab/STME) · [Torchvision RetinaNet](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/detection/retinanet.py)

| 方法 | 开发查准率 | 开发召回率 | 开发 F1 | 全部候选匹配上限 |
| --- | ---: | ---: | ---: | ---: |
| 七帧交付版 | 36.69% | 22.72% | 28.06% | 144/449 |
| 卷积循环时序，第 36 轮 | 35.63% | 6.90% | 11.57% | 111/449 |
| 双路差分时序，第 48 轮 | 31.87% | 12.92% | 18.38% | 119/449 |
| RetinaNet，第 8 轮 | 9.21% | 30.07% | 14.10% | 291/449 |
| 七帧交付版与 RetinaNet 候选融合 | 13.60% | 34.52% | 19.51% | 319/449 |

RetinaNet 提供的候选增加了覆盖范围，融合后的 155 个命中伴随 985 个误报。两路全部导出框的一对一匹配上限仍低于 80% 召回所需的 360/449。[覆盖明细](../../runs/snl-causal-study/prospective/recurrent-mixed-temporal-coverage.json) · [融合对照](../../runs/snl-causal-study/prospective/center-retinanet-agreement.json)

在同一两路候选上允许逐轴 −2、0、+2 px 平移和 0.8、1.0、1.2 倍缩放时，真值指导的一对一匹配上限为 **376/449**。据此训练局部复核网络。候选生成器仅使用编号 2–218 的训练标签，复核网络在其对 242–296 的预测上拟合，共 12,347 个训练图块。将该网络接到全训练段检测器后，校准首选的开发成绩为 25 个命中、35 个误报、F1 9.82%。学习到的框修正使候选覆盖从 319/449 降到 286/449。该试验表明真值指导的几何上限还未转化为可靠检测。[几何上限](../../runs/snl-causal-study/prospective/retinanet-bounded-geometry.json) · [复核网络对照](../../runs/snl-causal-study/prospective/retinanet-refinement-comparison.json)

追加了一次仿真对比度适配试验。每段在 SNL 早期训练标签的正局部对比度中抽取目标值，利用仿真掩膜边缘的渐变权重减弱过暗阴影。仿真对比度中位数由 0.2451 降至 0.0686，SNL 早期训练的正对比度中位数为 0.0353。复用直接训练对照，适配预训练后开发 F1 为 **18.24%**，配对差值为 −2.90 个百分点，95% 区间为 [−6.26，+0.28] 个百分点。全部候选匹配上限为 129/449。本轮源训练使用 CUDA，早先未经适配的源训练使用 CPU。[适配预训练报告](../../runs/snl-causal-study/prospective/simulation-pretraining-contrast-comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\experiment.py center-recurrent-temporal --profile prospective --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\experiment.py center-mixed-temporal --profile prospective --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\experiment.py retinanet --profile prospective --scale 3 --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_simulation_transfer.py --match-contrast --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_retinanet_refinement.py --output $run --ffmpeg $f
```

## 场景配准对照

对当前灰度图提取 SIFT 特征，经 RANSAC 估计到训练帧 296 的相似变换。将训练段每隔五帧的已验框中心变换到同一参考系，用最近训练目标点的距离过滤或减权。校准与开发段的全部图像均通过配准。开发段中，449 个目标处于训练点 30 像素内的比例从未配准时的 40.31% 增至 85.75%。在校准段选择距离和检测操作点后，七帧模型选择不加位置过滤。两路融合选择 75 像素过滤，开发段命中保持 155 个，误报从 985 减至 891，F1 从 19.51% 增至 20.74%。[配准与阈值明细](../../runs/snl-causal-study/prospective/registered-route-comparison.json)

另将视频配准后重训七帧中心网络。训练标签中心随坐标变换，框宽高按相似变换尺度缩放，预测框回到原雷达裁剪区域评分。直接匹配参考帧只成功配准 97/295 个训练帧，未配准帧使用原坐标。校准和开发帧全部匹配。开发段命中 68 个，查准率 29.96%、召回率 15.14%、F1 20.12%。[直接配准训练](../../runs/snl-causal-study/prospective/canonical-center-comparison.json)

相邻帧链式配准覆盖全部 295 个训练帧，训练框从 2,554 个变为视野内的 2,551 个。参考帧以后的图像只可借用已经处理的历史帧。该分支命中 78 个，查准率 21.97%、召回率 17.37%、F1 19.40%。两次配准重训均未通过验收。现有交付版重新通过了前缀因果性和图块坐标映射检查。正式盲测继续封存。[链式配准训练](../../runs/snl-causal-study/prospective/canonical-center-chained-comparison.json) · [前缀检查](../../runs/snl-causal-study/prospective/causality-check.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_registered_route.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_canonical_center.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_canonical_center.py --chain-registration --output $run --ffmpeg $f
```

## 难负样本排序与分数诊断

固定七帧网络、295 帧 SNL 训练集、种子 1729 和 48 轮训练时间表，在原分类损失上追加一个排序项。每个标注中心与同图块内最高分的 16 个背景峰比较，背景峰位于目标热图低于 0.05 的区域。仍由 14 帧校准段选择权重和操作点。

| 方法 | 开发查准率 | 开发召回率 | 开发 F1 | 原始候选一对一匹配上限 |
| --- | ---: | ---: | ---: | ---: |
| 同训练配置的直接训练对照 | 47.66% | 13.59% | 21.14% | 131/449 |
| 追加难负样本排序 | 47.52% | 14.92% | 22.71% | 130/449 |

F1 差值为 +1.57 个百分点，六帧分块配对的 95% 区间为 [−2.48，+5.17] 个百分点。本次排序试验未提供稳定增益，开发候选覆盖也没有增加。[配对结果和权重记录](../../runs/snl-causal-study/prospective/peak-ranking-comparison.json)

另固定七帧交付版做分数诊断。53 个训练抽样帧的全部原始候选可匹配 285/462，校准段为 32/112，开发段为 144/449。即使在开发段回溯选择阈值和 NMS，F1 也只达到 28.50%。回溯结果仅用于诊断，开发和封存验收仍用原协议。在本次检查的操作点中，校准、开发都没有达到 80% 查准率的配置。[分数与候选诊断](../../runs/snl-causal-study/prospective/released-score-separability.json)

```powershell
& $studyPy deliverables\snl-causal-study\experiment.py center-peak-ranking --profile prospective --output $run --ffmpeg $f
```

## 稠密光流候选对照

使用 OpenCV 4.12.0 的 DIS 中档实现，由当前帧向过去第 3、6 帧计算稠密光流。各自去除全图位移中位数后换算为像素每帧，两种间隔的速度幅值取几何平均。固定同一组 20 种矩形尺寸，比对局部暗度、相对运动及其乘积。另以框内平均光流给七帧模型的既有候选重打分，操作点与光流权重均在 14 帧校准段选择。[OpenCV DIS](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html)

开发复核共 449 个真值框。每帧允许 256 个候选且忽略分数门槛后，局部暗度、相对运动和联合线索的最高匹配数分别为 3、0、1。两种含光流的候选与七帧模型的候选合并后，覆盖上限仍为 144/449。对既有框重打分时，校准首选为光流权重 0，复现原 102 个命中与 176 个误报。保留原七帧模型作为交付版。[对照指标](../../runs/snl-causal-study/prospective/flow-candidate-study.json) · [逐帧计数](../../runs/snl-causal-study/prospective/flow-candidate-development.csv)

```powershell
& $py deliverables\snl-causal-study\snl_flow_candidates.py --output $run --ffmpeg $f
```

## 密集预测的候选损失

用现有七帧权重解码整个预测网格，并逐帧复现归档候选。在开发段，保留分数至少为 0.005 的所有格点时，借助真值做一对一匹配可覆盖 360/449 个目标。只保留局部峰值时降至 148/449，随后每帧取前 256 个则为 144/449。校准段上述三步的同口径覆盖分别为 86/112、33/112、32/112。[逐阶段诊断](../../runs/snl-causal-study/prospective/center-decode-audit.json)

取消局部峰值过滤、按原分数每帧保留 1,024 和 4,096 个预测，开发段的原始候选覆盖分别升至 292/449 和 319/449。两组都由原校准段选定操作点，都得到 107 个命中和 224 个误报，开发 F1 为 27.44%。大量形状可匹配的候选仍需可靠的质量估计与去重。[密集候选对照](../../runs/snl-causal-study/prospective/center-dense-candidate-ablation.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_decode_audit.py --output $run --ffmpeg $f
```

## 候选框定位质量训练

第一组用 295 帧公开训练标注联合训练七帧网络 48 轮。每个真值周围的 5×5 网格接受 IoU 分数监督，并按分类分数和 IoU 共同选择最多九个候选来拟合框几何。分别比较局部峰值解码与每帧保留分数最高的 1,024 个格点。组内检查点和阈值均由校准段选定。第二组固定原七帧网络和同一批密集候选框，只训练一个局部图块质量评分网络。评分训练用早期检测器对 242–296 帧的预测，该检测器仅在 2–218 帧拟合。校准段选中的分数为原分和局部质量分各占一半。[任务对齐学习](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)

| 方法 | 开发查准率 | 开发召回率 | 开发 F1 | 候选匹配上限 |
| --- | ---: | ---: | ---: | ---: |
| 七帧交付版 | 36.69% | 22.72% | 28.06% | 144/449 |
| 七帧密集候选，原分数 | 32.33% | 23.83% | 27.44% | 292/449 |
| 联合训练，局部峰值 | 24.06% | 7.13% | 11.00% | 92/449 |
| 联合训练，密集候选 | 41.73% | 11.80% | 18.40% | 288/449 |
| 固定密集候选，学习质量分 | 33.63% | 16.93% | 22.52% | 292/449 |

固定框的评分对照中，质量分让误报减少 74 个，同时少命中 31 个。相对使用同一组候选的原分数，开发 F1 下降 4.92 个百分点，六帧分块配对的 95% 区间为 [−7.98，−1.20] 个百分点。两个训练方法均未达到开发验收门槛。七帧交付版仍为首选，正式盲测保持封存。[联合训练结果](../../runs/snl-causal-study/prospective/task-aligned-comparison.json) · [固定框评分对照](../../runs/snl-causal-study/prospective/dense-quality-comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\experiment.py center-task-aligned --profile prospective --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_dense_quality.py --output $run --ffmpeg $f
```

## 信息几何阴影检测与七帧模型融合

按 Yan 等人的方法实现高斯邻域、Jensen–Shannon 散度、灰度与散度的二维最小交叉熵阈值、连通域提取和 Sobel 边缘过滤。将背景增强拆成配准历史中值、两分量高斯混合加权的滚动低秩背景，以及在最近五帧残差上追加 ADMM 分解三组。这里用项目已有的相位配准，低秩基由最近 6、12、18、24、30 帧的配准图像重估。原文的 SE-BSFV 用 SURF 与递推子空间更新，且原检测方法在每 20 帧结束时可回填当前片段的早期结果。本试验只输出当前框，轨迹确认只查过去 20 帧，允许跨最多 5 帧关联。二维阈值要求至少 70% 像素进入交叉熵计算，避免目标类和背景类都退化。论文像素高斯参数的文字与公式口径有歧义，分别试了把 `1e-5` 作为方差和标准差。[检测论文](https://doi.org/10.1109/JSTARS.2025.3568810) · [SE-BSFV 预处理](https://arxiv.org/abs/2501.09341)

在 14 帧校准段选择候选操作点。两种融合分别将候选补进七帧检测结果并去重，以及依据框内残差暗度调整七帧候选分数。开发复核沿用 48 帧、449 个框和逐帧一对一 IoU ≥ 0.50。前缀重跑逐项核对候选、分数、融合特征和轨迹支持计数。

| 开发方法 | 命中 | 误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 七帧交付版 | 102 | 176 | 36.69% | 22.72% | 28.06% |
| 原灰度信息几何对照 | 2 | 1,913 | 0.10% | 0.45% | 0.17% |
| 配准历史中值分支 | 1 | 1,489 | 0.07% | 0.22% | 0.10% |
| 滚动低秩分支 | 0 | 1,732 | 0.00% | 0.00% | 0.00% |
| 滚动低秩与 ADMM 分支 | 1 | 14,633 | 0.01% | 0.22% | 0.01% |
| 补入连通域候选 | 102 | 446 | 18.61% | 22.72% | 20.46% |
| 用阴影响应重新打分 | 74 | 71 | 51.03% | 16.48% | 24.92% |

仅扫描七帧模型的原分数阈值，校准首选仍为 0.25。重新打分的 F1 下降 3.14 个百分点，六帧配对分块重采样的 95% 区间为 [−5.18，−1.47] 个百分点。即使让真值在全部通过连通域几何及边缘检查的候选中挑选，单个实验分支也只能把七帧原始候选覆盖从 144/449 提到最多 148/449。该实现作为研究对照保存，交付继续使用原七帧模型。结果对应本地预处理、评测标注和因果协议。[运行结果](../../runs/snl-causal-study/prospective/information-geometry-study.json) · [逐帧统计](../../runs/snl-causal-study/prospective/information-geometry-development.csv)

```powershell
& $studyPy deliverables\snl-causal-study\snl_information_geometry.py --output $run --ffmpeg $f
```

## 配准残差与运动累积

训练样本和开发样本的局部阴影强度不同。按框外 4 像素环带中位灰度减去框内中位灰度，训练段每 5 帧抽样的 510 个框中位数为 9 级，242–296 帧为 7 级，校准和开发段均为 5 级。开发段 96.4% 的框具有正暗度差。单看框内平均标准化暗度，区分开发真值框与周围错开 12–23 像素的同尺寸框的 ROC AUC 为 0.916。该对照衡量真值附近的信号，整图虚警仍用检测评测统计。[通道对照](../../runs/snl-causal-study/prospective/motion-integration/channel-contrast-audit.json)

在相同中心检测器上使用未重采样的当前图、回投到原网格的标准化暗度、短期与长期配准历史残差，以及按 3 档速度和 16 个方向累积的暗化响应。295 帧训练 48 轮，每 12 轮记录检查点，由 14 帧校准段选第 24 轮。该操作点在开发复核命中 8 个、误报 25 个，查准率 **24.24%**、召回率 **1.78%**、F1 **3.32%**。其原始候选最多覆盖 111/449 个目标，与原七帧候选合计覆盖上限为 185/449。用相同起点的 10 帧和 15 帧前缀重跑，前 10 帧的 7 个输入通道逐字节一致。[训练与检查点记录](../../runs/snl-causal-study/prospective/motion-integration/comparison.json) · [特征前缀检查](../../runs/snl-causal-study/prospective/motion-integration/feature-prefix-check.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_motion_integration.py --output $run --ffmpeg $f
```

## 多尺度局部阴影候选

对标准化暗度与配准残差扫描多种矩形尺寸。以早期每两帧抽样的 148 帧拟合 LightGBM 评分和框偏移，共 771,816 个候选，其中 6,228 个与训练真值的 IoU ≥ 0.50。所有阈值和融合权重均在 14 帧校准段选择。开发段沿用 449 个社区标注框。

| 方法 | 开发命中 | 开发误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 七帧交付版 | 102 | 176 | 36.69% | 22.72% | 28.06% |
| 局部候选及学习框偏移 | 66 | 377 | 14.90% | 14.70% | 14.80% |
| 局部候选与稠密七帧候选融合 | 105 | 316 | 24.94% | 23.39% | 24.14% |

逐帧允许真值选择时，局部候选经框偏移后覆盖 275/449，七帧网络每帧 1,024 个格点覆盖 292/449，两路全部候选合计覆盖 348/449。实际融合只使用局部分数前 1,024 个框，参与融合的候选覆盖 323/449，最后准确选出 105 个。全部候选覆盖和实际检测分别报告。[局部候选结果](../../runs/snl-causal-study/prospective/local-shadow-proposals/comparison.json) · [融合结果](../../runs/snl-causal-study/prospective/local-shadow-proposals/dense-neural-fusion.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_local_shadow_proposals.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_local_shadow_proposals.py --output $run --ffmpeg $f --fusion
```

## 扩大候选后的留段评分

每帧保留七帧网络的 4,096 个格点，并加入全部多尺度局部暗度候选。用只在 2–218 帧训练的较早七帧模型生成 242–296 帧的候选，在这 55 帧的 485,970 个候选上拟合评分和框偏移，其中 3,792 个达到 IoU ≥ 0.50。评分同时读取框内暗度与残差、两路当前帧的局部高分，以及配准后的滞后 1、3、6 帧置信度。校准和开发使用已交付模型权重生成格点，按校准段选择修框和操作点。历史评分特征通过前缀一致性核对。[训练配置、权重哈希与评分](../../runs/snl-causal-study/prospective/forward-dense-fusion/comparison.json)

| 方法 | 开发命中 | 开发误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 格点七帧原分数 | 107 | 224 | 32.33% | 23.83% | 27.44% |
| 留段训练的两路评分与修框 | 116 | 403 | 22.35% | 25.84% | 23.97% |
| 两路评分与框投票 | 124 | 398 | 23.75% | 27.62% | 25.54% |
| 按帧分组的 LambdaRank 评分与修框 | 130 | 387 | 25.15% | 28.95% | 26.92% |
| LambdaRank 评分与框投票 | 114 | 242 | 32.02% | 25.39% | 28.32% |

七帧格点原始覆盖为 319/449，两路合并为 365/449，修框后为 361/449。上述覆盖由真值辅助的一对一匹配算出。固定评分排序后，未抑制时每帧前 40 个最多命中 99 个。非极大值抑制后前 40 个最多命中 183 个，所有剩余框最多命中 250 个。选择及去重须同时解决定位与误报问题。当前仍使用原七帧模型交付，两套正式盲测继续封存。[排序与候选损失](../../runs/snl-causal-study/prospective/forward-dense-fusion/ranking-audit.json)

在固定评分上比较硬抑制、Gaussian Soft-NMS 与按置信度加权的框坐标投票。校准段选中框投票，开发段多命中 8 个并少 5 个误报，F1 提高 1.57 个百分点。该结果仍低于交付七帧模型的 28.06% F1。[去重对照](../../runs/snl-causal-study/prospective/forward-dense-fusion/suppression-comparison.json)

另在同一批候选及留段训练数据上，将逐框分类换成按帧分组的 LambdaRank，以 IoU 分档监督前排候选，复用固定回归器。相对两路逐框评分，开发段多命中 14 个、少误报 16 个。校准段查准率为 23.35%、召回率为 34.82%，开发段与校准段均未达到双 80%。[按帧排序结果](../../runs/snl-causal-study/prospective/forward-dense-frame-ranking/comparison.json)

在 LambdaRank 分支上重新用校准段选择抑制方法和阈值，框投票的开发 F1 达到 28.32%，比原七帧交付版高 0.26 个百分点，六帧配对分块重采样的 95% 区间为 [−4.36，+4.68] 个百分点。该操作点在校准段的查准率和召回率为 31.97% 和 34.82%。[重选去重法](../../runs/snl-causal-study/prospective/forward-dense-frame-ranking/suppression-comparison.json) · [逐帧配对复算](../../runs/snl-causal-study/prospective/forward-dense-frame-ranking/paired-vs-release.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_forward_dense_fusion.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_forward_dense_fusion.py --output $run --ffmpeg $f --audit-ranking
& $studyPy deliverables\snl-causal-study\snl_forward_dense_fusion.py --output $run --ffmpeg $f --rank-by-frame
& $studyPy deliverables\snl-causal-study\snl_forward_dense_fusion.py --output $run --ffmpeg $f --rank-by-frame --audit-ranking
```

## 阴影存在评分与框定位评分

沿用 4,096 格点与局部候选的合并池、LambdaRank 置信度和已保存的框偏移。将候选评分拆成两项。第一项判断候选与任一真值的 IoU 是否达到 0.20。第二项仅在这样的近邻候选中学习 IoU ≥ 0.50。训练样本先在 242–296 帧内做三块留段修框，再按修后的坐标读取框内外灰度、暗度和残差，形成边缘两侧的对比特征。校准与开发使用固定的完整修框器。两项打分的混合权重和去重操作点只在 14 帧校准段选择。

另让原置信度决定哪些候选组进入输出，再用定位评分选择组内的一个现有框。两种流程沿用逐帧一对一 IoU ≥ 0.50，并回放原 LambdaRank 框投票结果作为对照。

| 方法 | 开发命中 | 开发误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 LambdaRank 与框投票 | 114 | 242 | 32.02% | 25.39% | 28.32% |
| 原分数混合阴影存在分 | 110 | 241 | 31.34% | 24.50% | 27.50% |
| 原分数混合存在分与定位分的乘积 | 102 | 177 | 36.56% | 22.72% | 28.02% |
| 保留组置信度，按定位分选代表框 | 111 | 381 | 22.56% | 24.72% | 23.59% |
| 保留组置信度，按目标分组训练的 LambdaRank 选框 | 105 | 316 | 24.94% | 23.39% | 24.14% |

修框后的候选池最多可匹配 361/449 个目标。只混合存在分时，校准选中的去重配置保留最多覆盖 260/449 个目标的候选。加入定位分时，该覆盖为 245/449。实际计数仍受置信度筛选约束。[逐项对照与权重](../../runs/snl-causal-study/prospective/box-quality/comparison.json)

另做组内排序诊断。开发段有 359 个真值同时拥有 IoU ≥ 0.50 的合格候选和 IoU 介于 0.20 至 0.50 的相邻候选。借助真值定义比较组，原置信度将合格框排在首位 138 次，单独定位分为 103 次，存在分与定位分的乘积为 113 次。此处用真值划组，仅诊断局部评分。[组内排序与实际检测](../../runs/snl-causal-study/prospective/box-quality/cluster-comparison.json)

按训练真值分组的 LambdaRank 使用 440 组、31,533 个留段候选，将同一目标的候选按 IoU 分档排序。应用时用原检测分选择重叠候选组，再用新分数选代表框。上述 359 组中有 114 组把合格框排在首位，低于原评分的 138 组。实际开发 F1 为 24.14%，相比原框投票下降 4.19 个百分点，六帧配对分块重采样的 95% 区间为 [−6.44，−2.13] 个百分点。[按目标排序结果](../../runs/snl-causal-study/prospective/local-box-ranking/cluster-comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_box_quality.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_box_quality.py --output $run --ffmpeg $f --clusters
& $studyPy deliverables\snl-causal-study\snl_box_quality.py --output $run --ffmpeg $f --local-rank
```

## 配准后的历史轨迹支持

固定已保存的两路候选、框偏移和 LambdaRank 基础置信度。将滞后 2、4、6 帧的七帧检测器分数图配准到当前帧，在静止假设和 3 档速度、12 个方向上查询每个当前框的历史支持。比较三种分数。独立历史分在每一帧单独选最强证据。一致轨迹分要求各帧共享一个速度向量。运动增益分使用一致轨迹分减去静止分。所有权重、输出阈值和去重方式均在 14 帧校准段选择。对同一前缀增删后续帧，历史分数逐项一致。

| 方法 | 开发命中 | 开发误报 | 查准率 | 召回率 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LambdaRank 与框投票 | 114 | 242 | 32.02% | 25.39% | 28.32% |
| 独立历史分 | 123 | 303 | 28.87% | 27.39% | 28.11% |
| 一致轨迹分 | 86 | 159 | 35.10% | 19.15% | 24.78% |
| 运动增益分 | 114 | 255 | 30.89% | 25.39% | 27.87% |

校准段在三个时序分支中选中一致轨迹分。开发段的 F1 差为 −3.54 个百分点，六帧配对分块重采样的 95% 区间为 [−6.05，−1.26] 个百分点。对开发段中与真值 IoU ≥ 0.20 的 26,176 个候选做诊断，以 IoU ≥ 0.50 为正例，原评分 ROC AUC 为 0.656，一致轨迹分单独为 0.542。历史证据在这次对照中的定位区分力较弱。[指标与逐项配置](../../runs/snl-causal-study/prospective/trajectory-support/comparison.json)

对照配置的 356 个输出框按 IoU ≥ 0.50 可匹配 114 个目标。保持中心并允许借用真值宽高时最多匹配 133 个。以预测中心距目标中心不超过 5 像素单独匹配，则覆盖 171 个。去重及框投票保留的全部 8,218 个候选在这两个诊断中的覆盖分别为 309 和 344 个。由此可区分尺寸误差和置信度筛选造成的候选中心损失。正式得分仍用 IoU ≥ 0.50。[框几何诊断](../../runs/snl-causal-study/prospective/trajectory-support/geometry-audit.json)

固定为原模型在每帧的输出数量，原模型在校准段命中 39 个，三种历史分均命中 37 个。三种历史分在开发段分别命中 111、115、112 个，对照为 114 个。等数量对照也未显示可复核的增益。[等数量对照](../../runs/snl-causal-study/prospective/trajectory-support/fixed-budget-comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_trajectory_support.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_trajectory_support.py --output $run --ffmpeg $f --geometry-audit
& $studyPy deliverables\snl-causal-study\snl_trajectory_support.py --output $run --ffmpeg $f --matched-budget
```

## 固定候选上的中心评分

训练样本沿用 242–296 帧的三块留段修框。将修框后中心距真值中心不超过 5 像素的候选标为正例，按每个目标的正候选数量校正权重。在同一中心周围采样固定网格的灰度、局部暗度、标准化残差和运动累积分数，并读取邻近分数图和历史证据。一共拟合 485,970 个候选，其中 19,888 个为正例。框偏移、候选来源和正式一对一 IoU ≥ 0.50 的指标沿用原评测。

| 方法 | 开发命中 | 开发误报 | 查准率 | 召回率 | F1 | 5 像素内中心匹配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LambdaRank 与框投票 | 114 | 242 | 32.02% | 25.39% | 28.32% | 171 |
| 融合中心评分与框投票 | 128 | 321 | 28.51% | 28.51% | 28.51% | 189 |
| 融合中心评分与中心距离去重 | 114 | 262 | 30.32% | 25.39% | 27.64% | 171 |

在全部开发候选上，识别 5 像素内正中心的 ROC AUC 从 0.935 提高至 0.948。校准选中中心评分占 25%、框投票及 0.55 输出阈值。开发 F1 差为 +0.18 个百分点，六帧配对重采样的 95% 区间为 [−1.87，+2.53] 个百分点。[训练与评测记录](../../runs/snl-causal-study/prospective/center-rescore/comparison.json)

固定原模型每帧输出数量后，校准命中由 39 降至 38，开发命中由 114 增至 115，中心匹配由 171 增至 172。单独改中心距离去重的开发 F1 为 26.76%。所选阈值的召回提升伴随更多输出，等数量对照中尚未观察到稳定收益。[等数量及去重对照](../../runs/snl-causal-study/prospective/center-rescore/ranking-budget.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_center_rescore.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_center_rescore.py --output $run --ffmpeg $f --audit-ranking
```

## 配准背景图对照

用编号 110–218 的 37 帧建立局部标准化的背景图，取配准后逐像素的中位数和上四分位数。另取当前帧前 3、8、15、24、36、51、68 帧估计滚动背景。在相同的框投票结果上比较原有外观评分与追加两种背景残差的评分。模型都用 242–296 帧的三块留段候选拟合，共 6,645 个候选组，其中 296 个一对一匹配。登记的配准锚点是第 296 帧。截断第 310 帧之后的画面和变换，背景特征逐元素一致。两组均在 310–323 帧选中评分权重 0.25 和阈值 0.50。[校准与权重记录](../../runs/snl-causal-study/prospective/background-atlas/calibration.json)

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LambdaRank 与框投票 | 31.97% | 34.82% | 114 | 242 | 32.02% | 25.39% | 28.32% |
| 候选组外观评分 | 30.83% | 36.61% | 120 | 274 | 30.46% | 26.73% | 28.47% |
| 外观评分加配准背景 | 31.30% | 36.61% | 119 | 267 | 30.83% | 26.50% | 28.50% |

追加配准背景后，开发段比外观对照少 7 个误报、少 1 个命中。F1 差为 +0.03 个百分点，六帧分块配对重采样的 95% 区间为 [−0.41，+0.39] 个百分点。固定每帧的输出数量为原框投票配置的数量后，背景组命中 114 个，与原框投票相同，比外观组少 1 个。两组固定框的真值辅助匹配上限均为 243/449，低于达到 80% 召回所需的 360/449。校准及开发均未达标，正式验收仍处于封存状态。[指标与配对区间](../../runs/snl-causal-study/prospective/background-atlas/comparison.json) · [逐帧计数](../../runs/snl-causal-study/prospective/background-atlas/development.csv)

```powershell
& $studyPy deliverables\snl-causal-study\snl_background_atlas.py --output $run --ffmpeg $f
```

## 局部光流检测与跨帧候选传播

第一组在逐帧全局配准后，用半分辨率 DIS 光流估计当前帧与滞后 2、5、11 帧间的局部位移。正反向位移闭合误差为光流加权。原生灰度、局部暗度、两档历史暗化、局部运动、配准误差改善及传播暗度合成七个通道。输入特征回投到 660×720 网格，沿用 295 帧训练和 48 轮预算。校准首选第 24 轮，命中 26 个、误报 33 个，查准率 44.07%、召回率 23.21%。开发段命中 34 个、误报 109 个，查准率 23.78%、召回率 7.57%、F1 11.49%。该分支原始候选可匹配 110/449 个目标，与交付七帧模型的候选合并可匹配 184/449。去掉校准首帧之后的输入重新计算该帧特征，两次结果逐元素一致。[检查点与指标](../../runs/snl-causal-study/prospective/local-flow/comparison.json)

第二组固定两路检测器，从滞后 1 和 3 帧各取最多 64 个候选，分别用全局变换和附加局部光流传播到当前帧。校准段选择传播分数权重、是否要求当前帧候选支持与去重操作点。局部传播通过相同前缀的候选框和分数逐元素检查。开发仍按 IoU ≥ 0.50 一对一匹配。

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 | 原始候选匹配上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 两路合并 | 34.25% | 22.32% | 87 | 121 | 41.83% | 19.38% | 26.48% | 184/449 |
| 两路合并加全局传播 | 31.46% | 25.00% | 91 | 158 | 36.55% | 20.27% | 26.07% | 211/449 |
| 两路合并加局部传播 | 34.25% | 22.32% | 87 | 121 | 41.83% | 19.38% | 26.48% | 217/449 |

局部传播扩充了原始候选，校准选定的分数下开发输出的计数与两路合并对照相同。校准首选全局传播，实际增加 4 个命中和 37 个误报，开发 F1 低于七帧交付版的 28.06%。两组实验均未达到双 80%，继续封存正式时段。[传播结果及校准配置](../../runs/snl-causal-study/prospective/local-flow/propagation-comparison.json) · [逐帧计数](../../runs/snl-causal-study/prospective/local-flow/propagation-development.csv)

```powershell
& $studyPy deliverables\snl-causal-study\snl_local_flow.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_flow_propagation.py --output $run --ffmpeg $f
```

## 独立实测背景的仿真预训练

用 VSDFormer 发布的澳大利亚和卡塔尔原始带斑雷达序列作为独立背景。两段各 313 帧、25 fps、720×720，预训练从每段原生网格取 720 个 192×192 图块。背景色调按训练段灰度分位数映射，时间滞后按源帧率换算。以同样的 1,440 个早期 Eubank 图块为对照，并保留七帧网络随机初始化的基线。早期 Eubank 背景图块避开已有阴影标注。[视频来源与版本](research-materials.md#snl-论文标注与开源方法核验)

两组预训练各用 8 轮，注入的 2,219 个当前帧框哈希相同。宽高、运动和衰减的抽样依据为 SNL 训练段。独立序列的监督只覆盖本实验注入的目标。两组随后均在相同的 295 帧 Eubank 标注上训练 48 轮，沿用每 12 轮保存检查点的设置。模型和检测操作点只在 310–323 帧选定。[源数据和合成框校验](../../runs/snl-causal-study/prospective/external-background-transfer/calibration.json)

| 初始化 | 校准查准率 | 校准召回率 | 开发查准率 | 开发召回率 | 开发 F1 | 原始候选匹配上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 随机初始化 | 36.54% | 16.96% | 47.66% | 13.59% | 21.14% | 131/449 |
| 早期 Eubank 背景预训练 | 51.92% | 24.11% | 42.97% | 12.25% | 19.06% | 125/449 |
| 澳大利亚与卡塔尔背景预训练 | 33.33% | 17.86% | 32.26% | 6.68% | 11.07% | 104/449 |

独立背景组比同预算早期实测背景组少命中 25 个、少误报 10 个，F1 下降 7.99 个百分点。六帧分块配对重采样的 95% 区间为 [−12.29，−3.65] 个百分点。三个分支均未达到双 80%。[检测结果及区间](../../runs/snl-causal-study/prospective/external-background-transfer/comparison.json) · [逐帧计数](../../runs/snl-causal-study/prospective/external-background-transfer/development.csv)

复算原 4,096 个中心格点与多尺度局部候选的合并池，开发覆盖为 365/449。加入独立背景模型的候选后仍为 365/449，加入早期 Eubank 背景候选则增至 371/449。两种预训练均未增加校准段的候选覆盖，仍为 96/112。这是忽略置信度的真值辅助匹配结果。[候选覆盖审计](../../runs/snl-causal-study/prospective/external-background-transfer/candidate-coverage.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_external_background_transfer.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_transfer_coverage.py --output $run --ffmpeg $f
```

## 密集候选的一对一评分

固定两路密集候选及学习过的框偏移。逐框网络只读各框的灰度、暗度、运动特征和原 LambdaRank 分数。邻域网络再读取在中心位置及框尺寸上最近的 12 个候选，以相对位置、尺寸比和 IoU 传递信息。两组采用相同随机种子、参数规模和 24 轮训练预算。在训练帧内只为每个可匹配真值分配一个正候选，并以最大 IoU 作辅助监督。首轮使用编号 242–296 的 55 帧留段候选，共 485,970 个框。检查点、混合权重和去重方式均在 310–323 帧选择。[校准记录](../../runs/snl-causal-study/prospective/set-ranking/calibration.json)

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 | 去重后候选匹配上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原评分与框投票 | 31.97% | 34.82% | 114 | 242 | 32.02% | 25.39% | 28.32% | 243/449 |
| 逐框一对一评分 | 31.54% | 36.61% | 120 | 270 | 30.77% | 26.73% | 28.61% | 268/449 |
| 邻域一对一评分 | 32.00% | 35.71% | 117 | 259 | 31.12% | 26.06% | 28.36% | 260/449 |

两组校准均选择框投票。逐框组在开发中比原评分多命中 6 个，同时多 28 个误报。F1 差为 +0.28 个百分点，六帧分块配对重采样的 95% 区间为 [−1.41，+2.26] 个百分点。若固定检查点，重新在校准段选择无需去重的操作点，逐框组在开发中命中 8 个、误报 25 个，邻域组命中 82 个、误报 1,109 个。邻域组有 197 个输出虽与某个真值的 IoU ≥ 0.50，却无法在一对一匹配中再获得命中。[开发结果](../../runs/snl-causal-study/prospective/set-ranking/comparison.json) · [无去重诊断](../../runs/snl-causal-study/prospective/set-ranking/without-suppression.json)

扩大训练时，新增 123–160 和 183–218 两段候选，分别由只用 2–100 和 2–160 帧训练的七帧检测器产生。评分器累计拟合 129 帧、1,177,819 个候选框。校准和开发候选几何保持相同，校准首选仍用框投票。[新增时段、检查点及指标](../../runs/snl-causal-study/prospective/set-ranking-expanded/comparison.json)

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 扩充后逐框评分 | 35.58% | 33.04% | 108 | 172 | 38.57% | 24.05% | 29.63% |
| 扩充后邻域评分 | 33.60% | 37.50% | 116 | 256 | 31.18% | 25.84% | 28.26% |

扩充后的逐框组比 55 帧逐框组开发 F1 高 1.02 个百分点，六帧分块配对区间为 [−0.78，+2.82] 个百分点。校准段选中邻域组，其开发 F1 为 28.26%。两组在去重前仍共享 361/449 的候选上限，去重后分别可匹配 262 个和 255 个。均未达双 80%。[扩充前后配对区间](../../runs/snl-causal-study/prospective/set-ranking-expanded/paired-vs-55-frames.json) · [逐帧结果](../../runs/snl-causal-study/prospective/set-ranking-expanded/development.csv)

```powershell
& $studyPy deliverables\snl-causal-study\snl_set_ranking.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_set_ranking.py --output $run --ffmpeg $f --audit-without-suppression
& $studyPy deliverables\snl-causal-study\snl_set_ranking.py --output $run --ffmpeg $f --expanded-fitting
```

## 局部坐标与单帧缺标

逐框比较 62 张 LabelMe 参考图与对应的雷达帧，覆盖校准段 112 个框和开发段 449 个框。以目标为中心的 34×34 参考窗口分别有 93 和 384 个通过相关与搜索边界检查。开发段可测偏移的中位数为 0.60 个原生像素，90 百分位为 1.47 像素。所有可测框均小于 2 像素。固定交付模型的输出，按估计偏移平移这些框后，校准命中由 26 增至 27，开发命中由 102 降至 98。正式匹配沿用原始坐标。[坐标审计](../../runs/snl-causal-study/prospective/reference-target-alignment.json) · [逐框偏移](../../runs/snl-causal-study/prospective/reference-target-alignment.csv)

训练集的单帧缺标实验用前后帧关联、双向模板相关和当前暗度筛选候选。在早期训练的 72 帧中人为隐藏 183 个框，选中的规则恢复 33 个，另产生 5 个误报。在稍后的 18 帧训练抽样中，以相同规则恢复 6/50，另产生 1 个误报。将规则用于原 295 帧训练数据，得到 3 个待验证的补标候选，保留原有 2,554 个框。[规则及留段复查](../../runs/snl-causal-study/prospective/gap-supervision/recovery-calibration.json) · [候选框位置](../../runs/snl-causal-study/prospective/gap-supervision/proposed-training-boxes.csv)

| 随机种子 | 原训练集开发 F1 | 追加三个候选框的开发 F1 | 固定每帧输出数量后的命中数 原训练集／追加后 |
| --- | ---: | ---: | ---: |
| 1729 | 21.14% | 25.65% | 61／61 |
| 3271 | 22.22% | 15.73% | 78／59 |

种子 1729 时，补标模型按校准选择的操作点命中 104 个，比同种子对照多 43 个，误报也多 191 个。等数量输出时差异消失。第二个种子的结果在原操作点和等数量对照中均下降。现有证据不支持采用补标配置，校准和开发均未达到双 80%。[种子 1729 结果](../../runs/snl-causal-study/prospective/gap-supervision/comparison.json) · [种子 3271 结果](../../runs/snl-causal-study/prospective/gap-supervision-seed3271/comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\audit_spatial_alignment.py --output $run --ffmpeg $f --audit-targets
& $studyPy deliverables\snl-causal-study\snl_gap_supervision.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_gap_supervision.py --output $run --ffmpeg $f --seed 3271
```

## 静态背景高分误报训练

用训练段 129 帧的留段预测挖掘额外训练图块。候选中心距当前及附近帧的标注中心超过 18 像素，在附近四帧的配准图块相关中位数至少为 0.80。从符合条件的候选选出 47 对图块，每对同帧。一侧围绕最高分候选，另一侧在符合条件的候选中随机抽样，两组的原检测分中位数分别为 0.166 和 0.071。两组使用同一 295 帧标注、随机种子和 48 轮训练预算。[配对图块](../../runs/snl-causal-study/prospective/hard-background/paired-training-tiles.csv)

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 同帧随机背景 | 45.45% | 22.32% | 54 | 92 | 36.99% | 12.03% | 18.15% |
| 同帧高分背景 | 44.00% | 19.64% | 39 | 79 | 33.05% | 8.69% | 13.76% |

高分背景组少 13 个误报，也少 15 个命中，开发 F1 差为 −4.39 个百分点。六帧分块配对区间为 [−6.79，−2.20] 个百分点。两组均未达到双 80%。[训练和评测记录](../../runs/snl-causal-study/prospective/hard-background/comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_hard_background.py --output $run --ffmpeg $f
```

## 配准轨迹的局部位置、尺寸和速度

用训练段全部 2,554 个标注点拟合局部位置密度和框尺寸，关联相邻帧并从稳定的五帧轨迹窗口提取 161 个速度样本。动态评分在相邻位置采样训练速度，只查询过去第 2、4、6 帧的雷达检测分数。删去校准首帧后的输入，首帧三种特征逐元素一致。使用同一密集候选池和框投票，仅在校准段选择评分混合权重与阈值。[位置和运动校准](../../runs/snl-causal-study/prospective/route-context/calibration.json)

| 方法 | 校准 F1 | 开发查准率 | 开发召回率 | 开发 F1 | 等数量输出命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原密集评分 | 33.33% | 32.02% | 25.39% | 28.32% | 114 |
| 位置 | 34.60% | 31.58% | 24.05% | 27.31% | 114 |
| 位置和尺寸 | 34.36% | 33.44% | 23.61% | 27.68% | 114 |
| 加入局部方向线索的两条分支 | 33.33% | 32.02% | 25.39% | 28.32% | 114 |

校准首选位置分支。其开发 F1 比原评分低 1.02 个百分点，六帧分块配对区间为 [−1.91，−0.09] 个百分点。含方向的分支均在校准选中零权重。[开发评测及配对区间](../../runs/snl-causal-study/prospective/route-context/comparison.json)

另固定候选中心，用邻近训练框的尺寸统计调整宽高。替换原框和追加新尺寸框两条分支均在校准选中原尺寸。开发诊断中，追加尺寸框能把真值辅助匹配上限从 361 增至最多 372/449，框投票后最多 261/449。正式验收仍封存。[尺寸对照](../../runs/snl-causal-study/prospective/route-context/size-comparison.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_route_context.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_route_context.py --output $run --ffmpeg $f --audit-size
```

## 历史热图反馈与位移校正

从同一七帧检测检查点初始化两个八通道模型，新增通道权重置零。两组均在早期 295 帧内抽取三帧连续图块，使用 293 个可作为片段末帧的编号和相同的随机种子优化 18 轮。零反馈组的新增输入始终为零。反馈组使用自己上一帧输出的中心概率图，每个训练步骤有 20% 概率丢弃该历史输入。位移校正版用当前灰度、上一帧灰度及上一帧热图预测采样偏移。两种反馈均只在训练片段内展开，使用同一校准段选择检查点和阈值。前缀复跑逐元素核验 304–311 帧的原始输出。[历史热图的设计参考](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2342_ECCV_2020_paper.php) · [校准记录](../../runs/snl-causal-study/prospective/feedback-center-aligned/calibration.json)

| 方法 | 校准查准率 | 校准召回率 | 开发命中 | 开发误报 | 开发查准率 | 开发召回率 | 开发 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 等预算零反馈 | 26.80% | 23.21% | 96 | 217 | 30.67% | 21.38% | 25.20% |
| 直接反馈 | 32.43% | 21.43% | 62 | 94 | 39.74% | 13.81% | 20.50% |
| 位移校正后反馈 | 32.43% | 21.43% | 62 | 95 | 39.49% | 13.81% | 20.46% |

按每帧与零反馈组相同的输出数量比较，零反馈命中 96 个，两种反馈各命中 91 个。固定反馈模型的权重，推理时清空历史通道，直接反馈和位移校正模型分别命中 92 个、91 个。本轮未观察到历史热图提供稳定增益。正式盲测保持封存。[直接反馈结果](../../runs/snl-causal-study/prospective/feedback-center/comparison.json) · [位移校正结果](../../runs/snl-causal-study/prospective/feedback-center-aligned/comparison.json) · [等数量比较](../../runs/snl-causal-study/prospective/feedback-center-aligned/fixed-count.json)

```powershell
& $studyPy deliverables\snl-causal-study\snl_feedback_center.py --output $run --ffmpeg $f
& $studyPy deliverables\snl-causal-study\snl_feedback_center.py --output $run --ffmpeg $f --align-history
```

## 独立复标样本

从训练、校准和开发段分别按时间等距抽取 6、4、8 帧。复标包提供原生灰度 PNG、对应慢速雷达短片、帧号清单与 LabelMe 操作说明。目标帧与短片都已核对，样本避开两套封存时段。请独立绘制 `shadow` 框，难以判断的区域标为 `uncertain`。[18 帧复标包](../../runs/snl-causal-study/independent-review.zip)

收到复标 JSON 文件夹或 ZIP 后，用下列命令检查文件名和原生网格，再比较逐帧的一对一 IoU ≥ 0.50 和中心距 ≤5 像素。脚本读取经过哈希验证的交付模型预测，用相同的框分别对照两套标注。`agreement.csv` 按帧统计双方各自独有的框、不确定框、预测在两套标注下共同命中或仅单方命中的数量。`agreement.json` 按时间段汇总两套查准率和召回率。复标结果用于区分是否看见同一目标与框边界差异，正式验收仍沿用已有协议。

```powershell
& $py deliverables\snl-causal-study\snl_independent_review.py --output $run --ffmpeg $f --labels 'D:\path\review-json.zip'
```
