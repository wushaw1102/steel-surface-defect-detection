# 钢材表面缺陷检测 · YOLO11 微调与对照实验

在 NEU-DET 数据集上做的一个可复现的检测实验：**数据划分 → 微调 → 独立测试集评测 → 可视化**，
并用一组「从零训练」的对照，量化预训练权重在这个任务上到底贡献了什么。

单卡 RTX 3050 Laptop (4G)，两组实验各 60 epoch，总耗时约 40 分钟。

## 结果

在 180 张**未参与训练与调参**的测试集上评测：

| 实验 | mAP50 | mAP50-95 | Precision | Recall | 最佳 epoch |
|---|---|---|---|---|---|
| 预训练微调 YOLO11n | **0.7213** | **0.3876** | 0.612 | 0.692 | 52 |
| 从零训练 同结构同超参 | 0.6760 | 0.3345 | 0.669 | 0.594 | 59 |

![result](assets/result_grid.png)

> 绿框为标注真值，红框为模型预测（数字为置信度）。大缺陷框得准，细长划痕与麻点类容易漏。

### 逐类别 mAP50-95

| 类别 | 微调 | 从零 |
|---|---|---|
| crazing（龟裂） | 0.1593 | 0.1440 |
| inclusion（夹杂） | 0.4373 | 0.4067 |
| patches（斑块） | 0.6240 | 0.5723 |
| pitted_surface（麻点） | 0.4024 | 0.4050 |
| rolled-in_scale（氧化皮压入） | 0.3014 | 0.2334 |
| scratches（划痕） | 0.4012 | 0.2457 |

## 值得说的四个点

**1. 预训练的真正收益在收敛速度上，而不只是涨点。**
mAP50 提升 4.5 点、mAP50-95 提升 5.3 点；但更明显的是微调在第 52 轮就到达峰值并走平，
从零训练到第 60 轮仍在缓慢爬升。同样算力下，预训练能把「收敛」这件事提前结束。

**2. 提升集中在召回，代价是精度。**
Recall 0.594 → 0.692（+9.8），Precision 0.669 → 0.612（-5.7）。
这不是「全都变好」，而是模型更敢报弱目标、误报随之增加。
阈值应当按漏检 / 误检的实际成本来定，用默认 0.25 只是省事。

**3. 输入尺寸按数据真实分辨率定，而不是照抄 640。**
NEU-DET 原图只有 200×200，resize 到 640 是纯上采样：不增加信息，单 epoch 从 9s 涨到 40s。
改用 `imgsz=320` 后，60 epoch 的训练时间压到 3 分钟出头。

**4. 弱类差是数据问题，不是训练不够久。**
crazing 只有 0.159，最差。缺陷细长、边界模糊，框偏几个像素 IoU 就掉出阈值；
它与 scratches 纹理接近，在混淆矩阵里可以看到互相误判。
类别不均衡也在放大这个问题：训练集 inclusion 有 799 个实例，pitted_surface 只有 322 个。

## 快速开始

```bash
# 1. 环境
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install ultralytics

# 2. 取数据（~145 MB，仓库里不存数据集）
python scripts/fetch_data.py

# 3. 划分 + 训练 + 可视化
python scripts/prepare_dataset.py
python scripts/train.py --exp finetune     # 预训练微调
python scripts/train.py --exp scratch      # 从零训练对照
python scripts/predict_demo.py             # 输出 assets/result_grid.png
```

## 项目结构

```
scripts/
  fetch_data.py       克隆源数据集到 neu_src/
  prepare_dataset.py  三划分 + 类别分布统计，生成 dataset/neu_det.yaml
  train.py            微调 / 从零训练，训练后自动在 test 上评测
  predict_demo.py     GT(绿) vs 预测(红) 可视化
dataset/neu_det/      YOLO 格式（由脚本生成，不入库）
runs/
  finetune/           训练日志、曲线、混淆矩阵、权重
  scratch/
  finetune_test/      测试集上的 PR 曲线与混淆矩阵
  *_test_metrics.txt  测试集指标
assets/result_grid.png
```

## 实验设置

| 项 | 值 |
|---|---|
| 模型 | YOLO11n（微调用 COCO 预训练权重；对照组随机初始化） |
| 数据 | NEU-DET, 1800 张, 200×200, 6 类 |
| 划分 | train 1377 / val 243 / test 180（seed=42） |
| imgsz / batch | 320 / 16 |
| epochs | 60（patience=15） |
| 其他 | seed=42, 默认增强, AMP |

## 局限

- 单一数据集、单光照条件、200×200 实验室图像，指标不能外推到产线场景；
- 未做 k 折交叉验证，小样本下指标有 ±1~2 点的波动；
- 未涉及视频跟踪与多传感器融合。

## 许可

数据集版权归 NEU-DET 原作者所有，本仓库仅包含代码与评测结果。
