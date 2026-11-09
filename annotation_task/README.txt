复标任务：18 张（每类 3 张，取自 test 划分，seed=2026）

【怎么标】
双击 annotate.html（Chrome / Edge 打开，图片已内嵌，不用装任何东西）
鼠标拖框 → 按 1~6 选类别 → 回车下一张 → 画完点“导出 JSON”

【规则】
- 原标注没有放进这个目录；定规矩的那 4 张可以对照 dataset/neu_det/labels/test/ 看，
  其余 14 张不要去看
- 4 张看着原标注定规矩，其余 14 张盲标（这样一致率才有意义）
- 不追求完美：标得不准也是有效数据，这正是要测的东西

【后续命令】
python scripts/import_annotations_json.py annotation_task/annotations.json
python scripts/compare_annotations.py --exclude crazing_274,inclusion_287,pitted_surface_288,scratches_294 --markdown annotation_task/annotation_audit.md
python scripts/annotation_compare_figure.py
