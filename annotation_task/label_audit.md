## 标注质量审查

对 `train,val,test` 全部 **1800 张 / 4189 个框**跑了一遍结构性检查（`scripts/audit_labels.py`，无需人工判读）：

| 检查项 | 判据 | 命中 |
|---|---|---|
| 框越过图像边界 | 角点超出 [0,1] | 0 处 |
| 退化框 | 宽或高 < 1% | 0 处 |
| 整图框 | 面积 > 95% | 56 处 |
| 同类重复框 | 同类且 IoU > 0.90 | 3 处 |
| 跨类重叠 | 异类且 IoU > 0.80 | 0 处 |
| 空标注图 | 无任何框 | 0 张 |

逐类框数：crazing 689，inclusion 1011，patches 881，pitted_surface 432，rolled-in_scale 628，scratches 548
