# Ka 频段视频 SAR 阴影仿真与检测

项目提供几何阴影生成、单通道合成孔径雷达（SAR）视频成像、阴影检测、单阴影跟踪、表观运动估计和独立测试。交付范围为仿真。实测、微波光子硬件链路和实验室 MAPS 验证依赖实验室资料。

[结题报告](deliverables/closeout/report.md) · [结题答辩稿](deliverables/closeout/defense.pptx) · [原申报书实验](deliverables/proposal/report.md) · [全 SAR 初步实验](deliverables/sar90/report.md)

[小阴影逐帧漏检诊断与复现](deliverables/miss-diagnostics/report.md)

## SNL 实测视频因果检测

使用公开的 SNL Eubank Gate 视频和逐帧车辆阴影框，比较背景差分、Faster R-CNN 和直接中心检测。每帧使用当前画面与历史信息。训练、开发与两套封存验收段按时间块记录，正式验收采用 IoU ≥ 0.50、查准率和召回率均达到 80%。[运行与验收流程](deliverables/snl-causal-study/README.md)

现有最佳配置在 48 帧开发复核中达到查准率 36.69%、召回率 22.72%、F1 28.06%。已完成 SIVED 预训练与开源 ByteTrack 对照，正式验收段仍处于封存状态。[分组结果](deliverables/snl-causal-study/README.md#sived-实测-sar-预训练对照) · [因果检测开发视频](runs/snl-causal-study/prospective/development-preview.mp4)

### 处理自己的雷达视频

安装项目神经网络依赖并确保 FFmpeg 可用。在仓库根目录运行以下命令，默认加载随项目附带的七帧研究模型。

```powershell
& .\.venv\Scripts\python.exe -m videosar detect-video --input 'D:\data\radar.mp4' --output-dir runs\my-video
```

打开 `overlay.mp4` 查看叠框，使用 `predictions.csv` 读取边框和置信度。`frames.csv` 包含所有已处理帧及其候选数。拼屏视频可加 `--crop X Y WIDTH HEIGHT` 指定雷达区域，输出坐标以该区域为基准。`video.json` 记录原视频帧号、裁剪、模型哈希和推理设备。模型在 Eubank Gate 开发段的成绩见上文。[复现参数及验证边界](deliverables/snl-causal-study/README.md#视频文件推理)

## 小阴影配对回波归因

固定场景、轨迹、散射体和孔径，使用 24 个场景族分别对比 300／600 MHz 和 25／10 dB。推理沿用冻结 U-Net、开源 SMP FPN 和背景差分。输出包括带配对校验的 96 段序列、逐帧预测、按场景族重采样的区间及模型失配案例。[实验结果与复现](deliverables/echo-attribution/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar echo-attribution --output-dir runs\echo-attribution-repro
```

## 独立孔径实验

固定 600 MHz、成像后 25 dB，在 24 个新场景族上比较 31、61、91 脉冲的居中孔径。输出包括静止点目标的方位响应、72 段仿真、冻结 U-Net 和开源 FPN 及背景差分预测、场景族配对区间与固定轮廓复算。[孔径实验报告](deliverables/aperture-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar aperture-study --output-dir runs\aperture-study-repro
```

## 孔径内目标运动对照

在 61、91 脉冲下比较逐脉冲移动与帧内固定两种目标运动模式，统一用同一真值轮廓评估小目标召回。可用 `--reference-run` 指定上一轮孔径数据，对逐脉冲移动组做逐字节复核。[运动对照报告](deliverables/aperture-motion-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar aperture-motion-study --reference-run runs\aperture-study --output-dir runs\aperture-motion-study-repro
```

## 小阴影局部对比度实验

在 600 MHz、成像后 25 dB 下独立生成 108 段纯回波数据，对比等预算续训和两个尺度的局部暗度输入。保留冻结 U-Net、仅调整操作点的 U-Net、开源 SMP FPN 和背景差分，统一在锁定测试集评价。[实验报告及复现](deliverables/contrast-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar contrast-study --output-dir runs\contrast-study-repro
```

## 小阴影中心候选实验

复用局部对比度实验的冻结训练和验证集，比较同一 U-Net 的分割输出与直接中心候选框。以 36 段新纯回波序列统一评价，保留冻结开源 FPN 和背景差分，并量化矩形轮廓能达到的 IoU 上限。[实验报告与复现](deliverables/center-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar center-study --training-run runs\contrast-study --output-dir runs\center-study-repro
```

## 跨帧弱候选比较

在 36 段独立纯回波上比较仅降阈值、基于轨迹的确认、三帧候选路径累积和低秩背景。验证集配置受无目标虚警约束，保留配置可由 `trajectory-detect` 复现。[实验报告](deliverables/trajectory-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar trajectory-study --output-dir runs\trajectory-study-repro
```

## 因果时序背景与小阴影加权

在原网格上比较继续微调、五帧历史中值差分和小阴影损失加权。保留原微调 U-Net 和 SMP FPN 的冻结检查点作为基线。先完成下文的小阴影实验及开源对照，再运行：

[因果时序对照报告](deliverables/temporal-study/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar temporal-study --dataset runs\small-shadow-repro\dataset --output-dir runs\temporal-study-repro
```

## 开源分割实现比较

先按下节生成小阴影训练与验证数据，再运行开源对照。使用 SMP 的 U-Net 和 FPN、Torchvision 的 LR-ASPP，并单独比较原网格与插值两倍输入的微调 U-Net。评估使用另外 36 段锁定回波序列，所有预测回到原始像素网格。

[开源实现对照报告](deliverables/open-compare/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar open-compare --dataset runs\small-shadow-repro\dataset --output-dir runs\open-compare-repro
```

## 小阴影改进实验

压力测试后另建 108 段纯回波序列。命令比较原配置、仅重选阈值和最小区域的配置，以及从原权重微调的时序 U-Net。输出含 48/24/36 段训练、验证和测试的分组清单、验证集阈值扫描、冻结记录和独立测试指标。

[小阴影改进实验报告](deliverables/small-shadow/report.md)

```powershell
.\.venv\Scripts\python.exe -m videosar small-shadow --output-dir runs\small-shadow-repro
```

## 纯回波压力测试

使用已有 U-Net 权重、概率阈值和区域筛选参数，与原背景差分基线对比。72 段新序列涵盖配对的速度、尺寸和静态暗区变化，以及独立的无目标场景。运行环境见下节。

```powershell
.\.venv\Scripts\python.exe -m videosar stress --output-dir runs\echo-stress-repro
```

程序在生成数据前写出 `protocol.json`，在正式推理前验证场景和掩膜。输出包含数据清单、预测、分组与配对指标，以及带对应帧号的失败案例。省略 `--model` 时加载申报书实验中的时序 U-Net 部署配置。

### 结题演示

上一步完成后，从仓库根目录运行以下命令。样例使用普通回波组的一段 20 帧序列。

```powershell
$py = '.\.venv\Scripts\python.exe'
$sample = 'runs\echo-stress-repro\dataset\stress-00-reference'
& $py -m videosar detect --input $sample --method learned --model deliverables\proposal\models\unet_temporal_post.json --output-dir runs\live-detect
& $py -m videosar track --input $sample --predictions runs\live-detect --output-dir runs\live-track
& $py -m videosar evaluate --input $sample --predictions runs\live-detect --tracks runs\live-track --match-iou 0.50 --output-dir runs\live-eval
```

查看 `runs/live-eval/preview.png` 和 `runs/live-track/tracks.csv`。验收复跑得到 20 帧全部匹配、零虚警，区域 IoU 为 0.864。

## 申报书路线

正式实验使用 120 段独立仿真序列、2,400 帧，按场景和轨迹分为 72 段训练、24 段验证和 24 段测试。图像域仿真占 80 段，代表性单通道 SAR 成像占 40 段。训练覆盖均匀、纹理、静态杂波背景及三档速度，测试另留六段道路背景。配准扰动只施加在观察帧，掩膜以固定地面坐标保存。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements-unet.txt
& $py -m videosar dataset --profile proposal --dataset-size 120 --seed 20261119 --output-dir runs\proposal-data-repro
& $py -m videosar study --dataset runs\proposal-data-repro --epochs 16 --device cpu --output-dir runs\proposal-study-repro
& $py -m videosar detect --input runs\proposal-data-repro\proposal-097 --method learned --model deliverables\proposal\models\unet_temporal_post.json
```

`study` 复用 [Segmentation Models PyTorch 的 U-Net](https://smp.readthedocs.io/en/latest/models.html#unet) 和 scikit-image 的亚像素配准。它对照单帧和相邻帧差分，逐项测噪声增强和后处理。`frozen_validation.json` 记录数据清单哈希、检查点哈希及验证集选定的操作点。最终目标级匹配使用 `IoU ≥ 0.50`，同时报告像素 F1、区域 IoU、每帧虚警和同机同尺寸的单帧平均耗时。U-Net 模型设置和权重需放在同一目录。

## 90 段全 SAR 初步实验

90 段、1,620 帧全部经过单通道 SAR 成像。按场景和轨迹划分，45 段训练、18 段验证、27 段独立测试。`--network context` 启用多尺度卷积检测，可同时训练单帧和因果时序版本。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m videosar dataset --profile sar --dataset-size 90 --seed 73 --output-dir runs\sar90-repro-data
& $py -m videosar benchmark --dataset runs\sar90-repro-data --network context --epochs 24 --seed 73 --output-dir runs\sar90-repro-results
& $py -m videosar detect --input runs\sar90-repro-data\scene-087 --method learned --model deliverables\sar90\models\causal_temporal.npz
```

每段包含观察强度、成像可见掩膜、几何掩膜和参数。`labels_by_scene.csv` 记录每段可见帧与前景像素数。`iou_intervals.json` 对测试序列成组重采样评估时序模型的 IoU 增益。已有权重旁的 JSON 提供验证集阈值，可用 `--threshold` 显式覆盖。

## 快速运行

在项目根目录使用 Windows PowerShell。已有可用的 `.venv` 时直接执行最后一行。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m videosar demo --config .\examples\urban-shadow.json --frames 16
```

程序打印新运行目录。打开其中的 `evaluate/preview.png`、`evaluate/metrics.json` 和 `track/tracks.csv`。预览叠层中青色代表测得轨迹，橙色代表短时外推。旧课程序列可用 `demo --scene legacy` 复查，也可把旧的 `outputs/day02` 作为 `detect --input`。

## 调参与分步调试

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m videosar --help
& $py -m videosar image --help
& $py -m videosar image --config .\examples\ka-simulation.json --bandwidth-hz 450000000 --output-dir runs\my-sar
& $py -m videosar detect --input runs\my-sar --method learned --model .\deliverables\models\causal_temporal.npz --output-dir runs\my-detect
& $py -m videosar track --input runs\my-sar --predictions runs\my-detect --max-gap 2 --output-dir runs\my-track
& $py -m videosar evaluate --input runs\my-sar --predictions runs\my-detect --tracks runs\my-track --output-dir runs\my-eval
```

- `simulate` 生成图像域序列。`image` 生成单通道 SAR 图像，并保存点目标校验和无阻挡参考。`demo` 用图像域序列跑完整链路。
- 参数优先级为 CLI、JSON、内置默认值。JSON 键用下划线，对应 CLI 的连字符。`--output-dir` 要求新目录，省略时自动在 `runs/` 下命名。`config.json` 保存配置，`run.json` 保存运行状态，`source.json` 记录输入目录及实际检测阈值。移动已有结果目录后，重新运行其下游步骤以更新输入记录。
- `detect` 和 `track` 需要 `frames.npy` 和 `metadata.json`。帧数组为有限非负强度 `frames[t,y,x]`。元数据至少提供正数 `frame_interval_s`，支持可选严格递增 `timestamps_s` 和正数 `pixel_size_m`。`evaluate` 另读同形状二值 `masks.npy`。学习权重旁的同名 JSON 保存验证集阈值。手动 `--threshold` 优先。
- `image` 同时保存 `geometric_masks.npy` 和 `unoccluded_power.npy`。几何掩膜取帧中心时刻，`masks.npy` 使用孔径覆盖及强度下降定义成像可见阴影。默认参数是仿真假设，可调整载频、带宽、脉冲间隔和每帧脉冲数。任意参数组合运行后，应检查 `point_check.json` 中的聚焦和距离峰指标。
- 轨迹 CSV 包含时间、像素中心、轨迹段及 `measured`、`imputed`、`missing`。速度使用测得点拟合。有地面尺度时另给出阴影表观速度，评估时在具有已知仿真几何的场景中估计目标运动。

## 从头复现实验

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -v
& $py -m videosar dataset --dataset-size 30 --seed 42 --output-dir runs\repro-data
& $py -m videosar benchmark --dataset runs\repro-data --seed 42 --epochs 16 --output-dir runs\repro-results
```

`dataset` 保存分组清单、每场景配置及标注。`benchmark` 保存训练权重、验证阈值扫描、测试总表、按场景及条件分组表、缺帧消融和测试预测。单独重训可运行 `train --dataset runs\repro-data --epochs 16`。项目在 Python 3.11.15、NumPy 2.4.6、Pillow 12.3.0 上验证。训练与推理使用 CPU，运行耗时随机器变化。
