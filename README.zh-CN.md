# 钢材表面缺陷检测 —— YOLO11 微调 vs. 从零训练

在 **NEU-DET** 上做的一个小而完整的检测实验：六类热轧钢材表面缺陷、1800 张灰度图、一块消费级显卡。
这个仓库不只报一个分数——它把**同一个结构、同一套超参跑两遍**，一遍从 COCO 预训练权重出发，
一遍从随机初始化出发，然后量化预训练权重到底值多少。

[English](README.md) · **简体中文**

![python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c?logo=pytorch&logoColor=white)
![ultralytics](https://img.shields.io/badge/ultralytics-8.4.208-0b7fab)
![task](https://img.shields.io/badge/task-object%20detection-4c8c4a)
![dataset](https://img.shields.io/badge/dataset-NEU--DET-8a6d3b)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## 目录

- [一步一步看（8 步，全图）](#一步一步看8-步全图)
- [实验结果](#实验结果)
- [逐类别拆解](#逐类别拆解)
- [这几个数字到底说明了什么](#这几个数字到底说明了什么)
- [快速开始](#快速开始)
- [流程](#流程)
- [仓库结构](#仓库结构)
- [数据集](#数据集)
- [实验设置](#实验设置)
- [关于复现](#关于复现)
- [局限](#局限)
- [后续计划](#后续计划)
- [致谢](#致谢)
- [许可](#许可)
- [FAQ](#faq)

---

## 一步一步看（8 步，全图）



### 1/8　数据：先看这任务有多难

![六类样本](assets/step1_samples.png)

六类缺陷、每类三张。低对比度、细结构、边界本身就模糊——这是后面所有指标的上限来源，
不是模型不够大能解决的问题。

![数据统计](assets/step1_stats.png)

训练集上顺手统计了三件事：类别不均衡（`inclusion` 799 个实例 vs `pitted_surface` 322 个）、
标注框面积的中位数只有 **11%**（**小目标为主**）、每张图通常 2–3 个框（最多 9 个）。

### 2/8　标注：先把训练目标审一遍

![标注框](assets/step2_annotations.png)

颜色即类别，阴影是被标注出来的缺陷区域，标题里给了该类框面积的中位数。

标注能不能直接拿来训练，是另一回事。所以动手之前先对标注做了一次**结构性审查**——
全部 1800 张 / 4189 个框，判据写死在 `scripts/audit_labels.py` 里，不依赖主观判断：

| 检查项 | 判据 | 命中 |
|---|---|---|
| 框越过图像边界 | 角点超出 [0, 1] | 0 |
| 退化框 | 宽或高 < 1% | 0 |
| 整图框 | 面积 > 95% | **56** |
| 同类重复框 | 同类且 IoU > 0.90 | **3** |
| 跨类重叠 | 异类且 IoU > 0.80 | 0 |
| 空标注图 | 一个框都没有 | 0 |

![标注审查](assets/step2b_label_audit.png)

结论很具体：**结构上是干净的（越界、退化、跨类重叠全为 0），但有 56 个框盖住了 95% 以上的画面**。
这类标注几乎不携带定位信息，模型只能学到"这里有缺陷"、学不到"在哪里"——这正好解释了
`pitted_surface` 那一类的 mAP@50-95 为什么上不去。另外 3 处是同一个缺陷被标了两遍。

`scratches` 的框宽只占图像约 6%，细长目标，是六类里最容易整个被漏掉的一类。

**所以干脆自己重标一遍。** 目的不是替换原标注，而是要量一量：这批标注到底把答案钉得有多死。
从 test 划分抽 18 张（每类 3 张）做第二遍标注：

- 其中 4 张是看着原标注定的规矩（crazing 按裂纹簇切开，inclusion 每条夹杂单独成框）；
- 剩下 14 张盲标，看不到原标注；
- 最后逐框比对，统计只算这 14 张。

标注是在本地跑的 **Label Studio**（社区版 1.23）里做的：一个项目、6 个矩形标签、18 张图导入，
标完直接导出 YOLO。`annotation_task/label_studio_export_yolo.zip` 就是那份导出原件，
`scripts/import_label_studio_export.py` 负责把 Label Studio 加的文件名 UUID 前缀去掉、
对回图像名。仓库里另外附了一个零安装的单页标注器 `annotate.html`，没有环境的人也能照着复现。

| 逐框（同类且 IoU ≥ 0.50） | 值 |
|---|---|
| 原标注框 / 我标的框 | 33 / 35 |
| 匹配上 | 13 |
| 召回 / 精度 / F1 | 0.394 / 0.371 / **0.382** |

这个数字很难看，第一反应是"标注有问题"——但并不是。逐框 F1 会重罚一件本身合理的事：
**同一片缺陷切成几个框**。所以把两边的框都栅格化成掩膜，改按面积比：

| 区域级（与切分粒度无关） | 值 |
|---|---|
| 区域 IoU | 0.579 |
| 原标注的缺陷区域被复标覆盖 | **0.820** |
| 复标区域里确实是缺陷的比例 | 0.663 |

82% 的缺陷区域被重标时又找到了；多出来的 34% 是框画得松。也就是说：分歧在框的松紧和切法，
不在漏看。

![复标对比](assets/step2c_annotation_compare.png)

这和上面的审查指向同一件事，只是从另一头看：在这份数据集上，"缺陷在哪"大致可以复现到
0.82 的区域召回，而"框该多大"并不可复现。对检测器来说，这意味着定位误差有一部分是标注
本身带进来的——第 5 节里 `pitted_surface` 的表现正是这个后果。

### 3/8　数据增强：模型实际看到的画面

![数据增强](assets/step3_augment.png)

只画 YOLO 默认开启的那几种（HSV、水平翻转、平移、缩放、mosaic）。
关键点：**框必须和像素同步变换**，框不跟着变，训练信号就是错的。

### 4/8　训练：两组实验放进同一个坐标系

![训练曲线](assets/curve_compare.png)

同结构、同超参、同 seed，唯一变量是初始权重。
微调组第 52 轮见顶后走平，从零组到第 60 轮仍在往上爬——这就是"预训练买到时间"的直接证据。

### 5/8　评测：只在没见过的 180 张图上

![逐类别](assets/per_class_map.png)

![混淆矩阵](assets/step5_confusion.png)

左边逐类对比微调与从零；右边是测试集上的混淆矩阵。
`crazing` 与 `scratches` 互相误判的那两条对角线，在矩阵里能直接看到。

![PR 曲线](assets/step5_pr_curve.png)

与其盯曲线猜，不如直接把阈值扫一遍（`python scripts/threshold_sweep.py`，跑在测试集上）：

| conf | Precision | Recall | F1 |
|---|---|---|---|
| 0.25 | 0.637 | 0.679 | 0.657 |
| 0.35 | 0.743 | 0.599 | 0.663 |
| 0.45 | 0.808 | 0.545 | 0.651 |

阈值从 0.25 提到 0.45：精度 **+17 点**，召回 **−13 点**，F1 几乎没有变化。
换句话说，工作点选在哪里不取决于哪个数字好看，而取决于这条线上**漏检和误报哪个更贵**。

### 6/8　同一批图：标注 vs 预测

![真值 vs 预测](assets/step6_quality.png)

上排是数据集自带的标注，下排是网络自己输出的框。这是"模型真的在预测，而不是在读答案"最直接的证据。

![批量结果](assets/result_grid.png)

扩大到 8 张图的批量结果（绿框真值、红框预测，数字为置信度）。

### 7/8　失效：它到底错在哪

![失效案例](assets/step7_failures.png)

四类错误由脚本**自动挑出**，不是手选好的一张：漏掉的小缺陷、漏掉的细长缺陷、
误报、以及定位不精确（匹配上了但 IoU 只有 0.50–0.62）。
每一条都能对应回上面的混淆矩阵。

### 8/8　批量推理：连续 24 帧

![批量推理](assets/step8_preview.gif)

顶部是帧号和当前置信度阈值。

以上所有图片，一条命令即可重建：

```bash
python scripts/workflow_figures.py --gif
```

---

## 实验结果

在 **180 张图 / 446 个实例的独立测试集**上评测——该划分既不参与训练，也不用于挑选 checkpoint：

| 实验 | mAP@50 | mAP@50-95 | Precision | Recall | 最佳 epoch（按 val） |
|---|---|---|---|---|---|
| **预训练微调** YOLO11n（COCO 权重） | **0.7213** | **0.3876** | 0.6120 | **0.6924** | 52 |
| **从零训练**（同结构，随机初始化） | 0.6760 | 0.3345 | **0.6695** | 0.5942 | 59 |
| Δ（微调 − 从零） | **+0.0453** | **+0.0531** | −0.0575 | **+0.0982** | — |

两组实验完全一致：60 epoch、`imgsz=320`、`batch=16`、`seed=42`、同样的增强流程。
唯一变量就是初始权重。

训练开销，直接取自 `runs/*/results.csv`：

| 实验 | 60 epoch 实际耗时 | epoch 中位数 | 第 1 个 epoch¹ | 模型规模 |
|---|---|---|---|---|
| 微调 | 566 s（9 分 26 秒） | 8.7 s | 41.9 s | 2.62 M 参数 |
| 从零 | 573 s（9 分 33 秒） | 9.1 s | 29.9 s | 2.62 M 参数 |

¹ 第一个 epoch 要付 RAM cache 填充与 CUDA autotune 的固定成本，不具代表性。

硬件：单卡 **RTX 3050 Laptop，4 GB**，`torch 2.6.0+cu124`，`ultralytics 8.4.208`。

---

## 逐类别拆解

测试集上的 mAP@50-95：

| 类别 | 微调 | 从零 | Δ | 相对变化 |
|---|---:|---:|---:|---:|
| crazing（龟裂） | 0.1593 | 0.1440 | +0.0153 | +10.6 % |
| inclusion（夹杂） | 0.4373 | 0.4067 | +0.0306 | +7.5 % |
| patches（斑块） | 0.6240 | 0.5723 | +0.0517 | +9.0 % |
| pitted_surface（麻点） | 0.4024 | 0.4050 | −0.0026 | −0.6 % |
| rolled-in_scale（氧化皮压入） | 0.3014 | 0.2334 | +0.0680 | +29.1 % |
| scratches（划痕） | 0.4012 | 0.2457 | +0.1555 | +63.3 % |

![逐类别 mAP](assets/per_class_map.png)

宏平均上的 +0.053 **并不是均匀分布的**。其中三分之二来自同一个类别：`scratches` 涨了 0.156
（相对提升 63 %），`rolled-in_scale` 涨了 0.068；`pitted_surface` 基本持平，而两组实验里都最差的
`crazing` 几乎没动。用均值概括这张表是不合适的——真正值得问的是**为什么**增益会集中在这两个类上。

---

## 这几个数字到底说明了什么

**1. 预训练买到的主要是收敛速度，而不只是精度。**
微调在第 52 轮就到峰值并开始走平；从零训练到第 59 轮还在缓慢爬升，此时预算已经用完。
差距是真的，但它有一部分是关于**时间**的陈述，而不是上限：如果从零训练也给 60 轮以上，
大概率能把差距追回一部分。预训练明确做到的事，是让搜索更早结束——在一块 4 GB 的笔记本显卡上，
这是"喝杯咖啡"和"耗掉一下午"的区别。

![验证曲线](assets/curve_compare.png)

**2. 增益在召回上，代价是精度。**
Recall 0.594 → 0.692（+9.8 点），同时 Precision 0.669 → 0.612（−5.7 点）。预训练模型更愿意
在弱证据上给出框，因此捞回了原本漏掉的细划痕，也在默认 `conf=0.25` 下带来更多误报。
这是一个要用"漏检成本 vs. 误报成本"来解的权衡，而不是挑 mAP 更好看的那一组。产线上这两种
成本很少相等。

**3. 输入尺寸应该由数据决定，而不是由习惯决定。**
NEU-DET 原图只有 200 × 200。上采样到 YOLO 默认的 640 不会增加任何信息——它只是插值出原本
不存在的像素——而每个 epoch 的计算量大致按缩放比例的平方增长。本仓库所有实验都用
`imgsz=320`（相对 200 已经是上采样），完整跑 60 epoch 控制在 10 分钟以内。任何固定小分辨率的
数据集都适用这条：抄 COCO 教程里的配置之前，先看源图尺寸。

**4. 弱类别是数据问题，不是"训练不够久"的问题。**
`crazing` 在两种设置下都停在 0.159。它的缺陷细长、对比度低、边界本身就模糊：框偏几个像素，
IoU 就掉到匹配阈值之下，这跟模型预测得多自信无关。`crazing` 与 `scratches` 的纹理也高度相似，
在 [`runs/finetune_test/confusion_matrix_normalized.png`](runs/finetune_test/confusion_matrix_normalized.png)
里能看到互相误判。类别不均衡进一步放大问题：训练集里 `inclusion` 有 **799** 个实例，
`pitted_surface` 只有 **322** 个（2.5 倍）。

**5. val→test 掉点是正常的，但先量化再解读。**
微调模型在它被选中的那个 checkpoint（第 52 轮）上，val 是 0.7357 / 0.4368（mAP@50 / @50-95），
test 是 0.7213 / 0.3876 ——因为 checkpoint 是在 val 上挑的。从零模型在第 59 轮上 val 是
0.6650 / 0.3543，test 是 0.6760 / 0.3345，它的 test mAP@50 反而略**高于** val，这是噪声而不是奇迹。
`pitted_surface` 在测试集只有 46 个实例，一个框排序错位就能让该类指标移动 ≈1/46 ≈ 0.02，
所以 2–3 点以内的差异不应被当作信号。

---

## 快速开始

上表的时间需要 CUDA 显卡；全部流程也能在 CPU 上跑，加 `--device cpu`（大约慢一个量级）。

```bash
# 1. 克隆
git clone https://github.com/wushaw1102/steel-surface-defect-detection.git
cd steel-surface-defect-detection

# 2. 虚拟环境
python -m venv .venv
.venv/Scripts/activate           # Windows；Linux/macOS 用 source .venv/bin/activate

# 3. 装与驱动匹配的 PyTorch（这里以 cu124 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. ultralytics（脚本还要用 matplotlib / pillow）
pip install ultralytics matplotlib pillow

# 5. 取数据（~145 MB，仓库里不存数据集）
python scripts/fetch_data.py

# 6. 生成 train/val/test 三划分，并写出 dataset/neu_det.yaml
python scripts/prepare_dataset.py

# 7. 跑两组实验（3050 Laptop 上各约 10 分钟）
python scripts/train.py --exp finetune
python scripts/train.py --exp scratch

# 8. 单独在测试集上评估任意 checkpoint
python -c "from ultralytics import YOLO; print(YOLO('runs/finetune/weights/best.pt').val(data='dataset/neu_det.yaml', split='test', imgsz=320))"

# 9. 重新生成本 README 里的全部图片（--gif 会多渲一段 24 帧动图）
python scripts/workflow_figures.py --gif   # step1~step8 全套
python scripts/predict_demo.py             # assets/result_grid.png
python scripts/make_figures.py             # assets/curve_compare.png、assets/per_class_map.png
```

第 5–7 步是幂等的：`fetch_data.py` 发现 `neu_src/` 已存在就跳过克隆，
`prepare_dataset.py` 每次都会清空并重建 `dataset/neu_det/`。

不想训练？微调后的 `best.pt`（5.2 MB）挂在
[release v1.0.0](https://github.com/wushaw1102/steel-surface-defect-detection/releases/tag/v1.0.0) 上。
放进 `runs/finetune/weights/`，或者直接指给可视化脚本：

```bash
# 也可以自己下
curl -L -o best.pt https://github.com/wushaw1102/steel-surface-defect-detection/releases/download/v1.0.0/best.pt
python scripts/predict_demo.py --weights best.pt
```

---

## 流程

```mermaid
flowchart LR
    A["fetch_data.py<br/>git clone NEU-DET<br/>（YOLO 格式镜像）"] --> B["prepare_dataset.py<br/>打乱 + 划分<br/>1377 / 243 / 180"]
    B --> C["dataset/neu_det.yaml<br/>+ 6 个类别名"]
    C --> D1["train.py --exp finetune<br/>yolo11n.pt（COCO）"]
    C --> D2["train.py --exp scratch<br/>yolo11n.yaml（随机）"]
    D1 --> E1["runs/finetune/<br/>best.pt、results.csv"]
    D2 --> E2["runs/scratch/<br/>best.pt、results.csv"]
    E1 --> F["model.val(split='test')<br/>180 张未见过的图"]
    E2 --> F
    F --> G["runs/*_test_metrics.txt"]
    E1 --> H["predict_demo.py<br/>assets/result_grid.png"]
    E1 --> I["make_figures.py<br/>曲线 + 逐类柱状图"]
    G --> I
```

三划分只在 `prepare_dataset.py` 里做一次（`seed=42`），结果落盘成文件，
因此两组实验看到的是逐字节相同的划分。

---

## 仓库结构

```
scripts/
  fetch_data.py        浅克隆源数据集到 neu_src/
  prepare_dataset.py   确定性三划分，写出 dataset/neu_det.yaml 与类别统计
  train.py             微调 / 从零训练，结束后自动在 test 上评测
  workflow_figures.py  重建 README 里的步骤图 step1~step8（含动图）
  predict_demo.py      8 张测试图的 真值(绿) vs 预测(红) 拼图
  make_figures.py      重建 assets/curve_compare.png 与 assets/per_class_map.png
  threshold_sweep.py   扫描不同置信度阈值下的精度/召回
  audit_labels.py      标注结构性审查（越界／重复／退化框）
  import_annotations_json.py  标注页导出的 JSON → YOLO 标签（左上原点 → 中心原点）
  import_label_studio_export.py  Label Studio 的 YOLO 导出包 → YOLO 标签（去掉文件名 UUID）
  compare_annotations.py      与原标注的逐框 + 区域级一致性比对
annotation_task/       复标任务：18 张图、标注页、以及复标结果
  annotate.html        单文件标注页，图片已内嵌，零安装
  label_studio_export_yolo.zip  Label Studio 导出的原始 YOLO 包
  annotations.json     标注页导出的原始结果
  labels/              同一份结果转成的 YOLO 标签
  annotation_audit.md  第 2 节引用的那些一致性数字
dataset/               由 prepare_dataset.py 生成，已 git-ignore
runs/
  finetune/            results.csv、results.png、PR/F1 曲线、混淆矩阵、权重
  scratch/             同上，从零训练对照
  finetune_test/       测试集上的 PR 曲线与混淆矩阵
  scratch_test/
  *_test_metrics.txt   上面各表引用的原始数字
assets/                本 README 内嵌的图
```

---

## 数据集

**NEU-DET**（东北大学热轧带钢表面缺陷数据库）：六类——`crazing`、`inclusion`、`patches`、
`pitted_surface`、`rolled-in_scale`、`scratches`。灰度、200 × 200、共 1800 张。
本仓库使用 YOLO 格式镜像
[`Marfbin/NEU-DET-with-yolov8`](https://github.com/Marfbin/NEU-DET-with-yolov8)，
这样就没有任何隐藏的格式转换步骤。**完整数据集不在本仓库分发**；只有复标实验用到的那 18 张
（每类 3 张）放在 `annotation_task/images/` 里，好让复标这一步可以原样重放。

| 划分 | 图片 | 实例 | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1 377 | 3 180 | 515 | 799 | 650 | 322 | 480 | 414 |
| val | 243 | 563 | 95 | 123 | 132 | 64 | 79 | 70 |
| test | 180 | 446 | 79 | 89 | 99 | 46 | 69 | 64 |

`val` 是从源 `train` 里按 15 %（`seed=42`）切出来的；源 `test` 划分原样保留，只用于最终数字。

---

## 实验设置

| 项 | 值 |
|---|---|
| 模型 | YOLO11n，2.62 M 参数 |
| 初始化 | COCO 预训练（`yolo11n.pt`） vs. 随机（`yolo11n.yaml`） |
| 输入尺寸 | 320 × 320（源图 200 × 200） |
| Batch / epochs | 16 / 60，`patience=15`（早停从未触发） |
| 优化器 | Ultralytics `optimizer=auto` → SGD + warm-up，`lr0=0.01` |
| 增强 | Ultralytics 默认（mosaic、HSV、翻转、缩放） |
| 精度 | AMP（fp16） |
| 随机种子 | 42 |
| 数据加载 | `cache="ram"`、`workers=2` |
| checkpoint 选择 | val mAP@50-95 最优 |

每次运行解析后的完整配置都提交为 `runs/<exp>/args.yaml`，因此本文里的表格可以追溯到真实产物，
而不是凭记忆复述。

---

## 关于复现

四个实际踩过、值得先知道的点：

**先划分，再调参。** `val` 用于挑 checkpoint 和早停；`test` 只在最后碰一次。如果开始对着 `test`
调超参，上面的表就失去意义了。

**`dataset/neu_det.yaml` 里是绝对路径。** `prepare_dataset.py` 会写入
`path: <dataset/neu_det 的绝对路径>`，因为 Ultralytics 是这样解析数据集根目录的。
该文件在本机生成且已 git-ignore——不要提交，否则所有人都继承你的目录结构。

**挪动项目位置后要重新生成 `neu_det.yaml`。** `path` 是生成时写死的，换目录就重跑一次
`prepare_dataset.py`。

**Windows 上 `workers>0` 启动很慢。** 每个 worker 都要重新 import torch；配合 `cache="ram"`
和只有 1377 张训练图，数据加载根本不是瓶颈，所以 2 个 worker 是启动成本和吞吐之间的合理折中。
除非你测出收益，否则不要往上加。

---

## 局限

- **单一数据集、单一成像条件。** 200 × 200 的实验室灰度小块描述不了产线：光照、尺度、运动模糊、
  类别先验都会不同。本文的绝对数字不能当作工业基准引用。
- **每个配置只跑了一个种子。** 没做 k 折交叉验证，因此几个点的差异要当作噪声（见上面第 5 点）。
- **没做结构搜索。** 只试了 YOLO11n；更大的变体很可能单靠容量就能把差距抹平，那样对本次消融的
  解释就会改变。
- **只有检测框。** 没有实例分割、没有缺陷面积回归、没有跨帧跟踪。
- **全程 `conf=0.25`。** 没有按类调工作点，而逐类结果表恰恰说明应该按类调。

---

## 后续计划

- [ ] 跑 3 个种子，给消融加上均值 ± 标准差
- [ ] 在 `val` 上扫逐类置信度阈值，再在 `test` 上报告
- [ ] 同 epoch 预算下对比 `yolo11s` / `yolo11m`，把"预训练有用"和"容量有用"分开
- [ ] 导出 ONNX 并测 CPU 延迟——目标设备是产线旁的工控机
- [ ] 针对稀缺的 `pitted_surface` 加 copy-paste 增强

---

## 致谢

- 数据集：**NEU-DET**，K. Song, Y. Yan, *A noise robust method based on completed local binary
  patterns for hot-rolled steel strip surface defects*, Applied Surface Science, 2013.
- YOLO 格式镜像：[`Marfbin/NEU-DET-with-yolov8`](https://github.com/Marfbin/NEU-DET-with-yolov8)。
- 检测器：[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)。

## 许可

本仓库代码以 [MIT License](LICENSE) 发布。NEU-DET 的图像与标注版权归原作者所有，
在搭建环境时下载，不在本仓库分发。

---

## FAQ

**NEU-DET 上 0.72 mAP@50 算好吗？**
对这个数据集、320 px 输入、nano 级检测器来说属于正常区间。文献数字只有划分、输入尺寸和评测
协议完全一致时才可比——这也正是本仓库把划分和原始 `results.csv` 一并提交的原因。

**为什么不只报微调模型？**
因为"微调有用"只有在知道**多少**和**在哪里**有用时才有意义。从零对照只花十分钟，却能把一句
断言变成一次测量——而逐类表显示增益集中在两个类上，单一数字会把这个信息藏掉。

**能换成我自己的六类数据吗？**
可以。改 `scripts/prepare_dataset.py` 里的 `CLASSES`，让 `fetch_data.py`（或你自己的拷贝步骤）
指向 YOLO 格式的数据即可，流程里没有其他地方绑死 NEU-DET。

**既然早停从没触发，配 `patience=15` 干嘛？**
它是为更大预算准备的保险。60 轮下从零训练到第 59 轮还在涨，一旦把轮数拉长，它立刻就有用了。
