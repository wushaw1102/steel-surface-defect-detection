复标任务：18 张（每类 3 张，取自 test 划分，seed=2026）

【用 Label Studio 标】
1. 启动服务（在项目根目录）
     .venv-ls\Scripts\label-studio.exe start
2. 浏览器打开 http://localhost:8080
   账号 
3. 项目 "NEU-DET re-annotation · 18 crops" ，18 张图导入，6 个类别配置
4. 按 1~6 选类别 → 鼠标拖出矩形 → Ctrl+Enter 提交并跳下一张
5. 项目页 Export → 格式选 YOLO → 得到 zip



【规则】
- 原标注没有放进这个目录。4 张用来定框选粒度（对照 dataset/neu_det/labels/test/），
  其余 14 张盲标，统计只算这 14 张
- 不追求完美：标得不准也是有效数据，这正是要测的东西

【后续命令】
python scripts/import_label_studio_export.py annotation_task/label_studio_export_yolo.zip
python scripts/import_annotations_json.py annotation_task/annotations.json
python scripts/compare_annotations.py --exclude crazing_274,inclusion_287,pitted_surface_288,scratches_294 --markdown annotation_task/annotation_audit.md
python scripts/annotation_compare_figure.py
