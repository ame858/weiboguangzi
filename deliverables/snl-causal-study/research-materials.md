# SNL 论文、标注与开源方法核验

以 Eubank Gate 原视频为共同参照。检索用论文全名、DOI、作者、视频名和 `annotation`、`code`、`weights`，沿 DSENet 表 3 查回原始方法，并查找 GitHub、数据存档及引用该场景的基准。优先检查论文的数据声明、补充材料和作者发布入口。

| 来源 | 已核验材料 | 能做的对照与缺口 |
| --- | --- | --- |
| Sandia 原视频与 Jiangxb8 社区仓库 | 原 MP4 已落盘，SHA‑256 为 `730ae5eb7bde0b80b889b25118042253031d03356ef6e5fc668116c873f12396`。社区版固定到提交 `28a6ee2cff7913d817816c01589e46c862927df0`，本实验读经 Git 对象哈希校验的 LabelMe `car` 矩形和附带的参考图。 | 开源 Faster R-CNN 已按项目时间划分重训与评分。源仓库的本地续训权重没有取得。 [原视频](https://www.sandia.gov/app/uploads/sites/124/2021/08/eubankgateandtrafficvideosar.mp4) · [固定版代码和标注](https://github.com/Jiangxb8/Video-SAR/tree/28a6ee2cff7913d817816c01589e46c862927df0) |
| DSENet，10.3390/rs18101623 | §3.1 描述 900 帧及按时间顺序划分为 7:1:1。数据声明给出 Sandia 原视频。图 2 有三帧示例，主表共有 690 个评测真值。 | 有图示可做早期帧对齐。未检索到可下载的逐帧标签、精确帧清单、官方推理代码或权重。 [论文、图 2、数据声明](https://doi.org/10.3390/rs18101623) |
| MambaShadowDet，10.3390/rs17020214 | 论文表 1 给出 SNL 上 1,694 个评测真值、88.90% 查准率和 73.26% 召回率。数据声明明确可向作者申请原始数据。 | 现有材料支持查表与阅读架构。未取得该表对应的逐帧标注、精确切分、官方代码或权重。 [论文、表 1 与数据声明](https://doi.org/10.3390/rs17020214) |
| ShadowDeNet、STCADeNet | 前者的数据声明指向同一 Sandia 原视频，另一个试验视频受版权限制。后者用跨帧特征与框级聚合。DSENet 表 3 的相关成绩引自这些论文。 | 可参考方法。未检索到足以在当前划分复跑的标注与权重。 [ShadowDeNet](https://doi.org/10.3390/rs14020320) · [STCADeNet](https://www.sciencedirect.com/science/article/pii/S0957417425015039) |
| VSMB / VSMT，10.1109/LGRS.2025.3592711 | 作者仓库列出同名 Eubank Gate 片段及多目标跟踪数据的公开入口。仓库展示压缩示例，百度网盘入口未取得可用归档，已有响应为链接不存在。 | 优先查作者镜像和帧清单。Hugging Face 发布请求尚开放。尚待核验原尺寸框、轨迹字段、时间覆盖与授权条件。 [论文](https://doi.org/10.1109/LGRS.2025.3592711) · [作者仓库](https://github.com/softwarePupil/VSMB) · [镜像请求](https://github.com/softwarePupil/VSMB/issues/1) |
| TalentBoy2333 的 VideoSAR-segmentation | 同一场景的开源 CNN-LSTM／ConvLSTM 分割训练与评价脚本。README 描述作者用 LabelMe 自行标注。 | 本轮未取得脚本依赖的全量 `dataset/segmentation` 或 `param/` 权重。社区框标注无法直接复现该项目的像素监督。 [仓库](https://github.com/TalentBoy2333/VideoSAR-segmentation) |
| SDM-Car / SDM-Car-SU | SDM-Car 提供光学卫星的弱小车辆序列、框及轨迹编号，64 段为作者划定的训练序列。SDM-Car-SU 在该数据上加入全图位移。两者为非商业学术研究用。 | 可检验独立序列的时序表征迁移。本机对公开归档的下载地址解析失败，尚未取得训练图像。光学车辆外观与 SAR 阴影存在模态差异。 [SDM-Car 作者仓库](https://github.com/TanedaM/SDM-Car) · [SDM-Car-SU 数据页](https://huggingface.co/datasets/AlexChenWhu/SDM-Car-SU) |
| SARDet-100K / SARATR-X-v2 | 前者公开静态 SAR 检测的框架、配置和模型发布入口。后者公开结构预训练代码，检查点需向作者申请。 | 可作为 SAR 空间特征初始化。两套来源提供的资料均未取得 SNL 运动阴影的训练标签。 [SARDet-100K](https://github.com/zcablii/SARDet_100K) · [SARATR-X-v2](https://github.com/waterdisappear/SARATR-X-v2) |

PIFTrack 作者另行发布了检测与跟踪代码，列出检测权重及 VSMB 跟踪输出的网盘入口。本轮访问未能打开两项下载页。取得文件后可作为外部方法对照，需核对 Eubank 帧映射和权重所用训练时段。[作者发布页](https://github.com/softwarePupil/PIFTrack)

MTT-RFBMFC 在 SNL 和 BIRM 的视频 SAR 上评测。固定提交 `db0004321fffb4ed5b1ec12c217bdd912c5ced43` 的公开文件树共 188 项，包含推理代码和标注转换程序。SNL 检查点由作者另行分享，访问到达提取码页，尚未取得权重和训练帧清单。复跑需要先核对其训练时段。[作者发布页](https://github.com/hongjianzhi/MTT-RFBMFC)

VSDFormer 固定提交 `c61fb8e744c06f22815f07ad1a0377bebd7c9feb` 提供多段雷达视频。已取得澳大利亚和卡塔尔的原始带斑视频，各 313 帧、25 fps、720×720。它们可提供与 Eubank 独立的实测背景。序列的真实阴影框仍待取得。[固定版视频目录](https://github.com/BFY-official/VSDFormer/tree/c61fb8e744c06f22815f07ad1a0377bebd7c9feb/Videos)

2026-11-23 复核 VSMB 的发布入口。作者仓库列出 45 段标注视频，并提供网盘和邮件申请入口。本轮仍未取得可读取的数据归档，Hugging Face 镜像请求仍开放。取得 VSMB 后，先核对每段实际采集来源并识别与 SNL 重叠的画面，再选择独立实测场景。[作者仓库](https://github.com/softwarePupil/VSMB) · [镜像请求](https://github.com/softwarePupil/VSMB/issues/1)

余安喜等的跟踪研究声明机载实测部分使用 SNL，逐帧阴影由人工标注，论文列出向通讯作者申请数据的方式。这可作为同源重标的另一条线索，需索取精确帧号和原尺寸框。何志华等发表了 SNL 车辆标注示例，期刊页面的资源附件数为零。这两项均未找到可直接计分的逐帧文件。[余安喜等，2025](https://doi.org/10.3390/rs17060959) · [何志华等，2022](https://doi.org/10.11999/JEIT210853)

## 可以直接复用的证据

- Jiangxb8 的 Faster R-CNN 在 14 帧校准段选中第 8 轮。在相同 48 帧、449 个标签的开发复核段命中 106 个、误报 536 个，查准率 **16.51%**、召回率 **23.61%**、F1 **19.43%**。七帧交付工具的对应数值为 36.69%、22.72%、28.06%。两者都用逐帧一对一框 IoU ≥ 0.50。 [开源方法实跑记录](../../runs/snl-causal-study/prospective/decision-jiang-faster-rcnn-1x-epoch8.json)
- DSENet 图 2 三个示意帧分别对应社区标注编号 3、91、145。脚本提取的 12 个黄色框中，10 个中心可在 10 像素内与社区框一对一配对，按 IoU ≥ 0.50 的匹配为 0。这支持复核示意框尺寸和标注定义，但三个图示帧不足以推断论文评测标签的整体一致性。 [逐帧对齐](../../runs/snl-causal-study/paper-reference/dsenet-figure2-label-agreement.json)

现阶段采用社区框做同口径对照。获得独立标注时，先确认视频散列或用图像内容核验帧映射，再核对框与掩膜的含义。在公开开发帧统计双方一对一匹配及单方标注，保存逐帧差异。与封存时段重叠的标签按正式验收协议保管。可向论文作者索取帧清单、框或掩膜文件、权重与推理入口。本轮只使用公开检索。
