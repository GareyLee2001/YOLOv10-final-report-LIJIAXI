# AT3② 复现说明（简要）

**学生：** 李嘉烯（D26092110161）  
**论文：** YOLOv10 (NeurIPS 2024)  
**实现：** Ultralytics `yolov10n.pt`  
**数据：** **coco128**（常用小集，约 128 张；比 coco8 更有代表性，又远小于全量 COCO）

## 1. 复现了什么

1. **Path A：** 官方预训练权重，在 coco128 上评测 + 推理可视化（不训练）  
2. **Path B：** 同数据上微调 **100 epoch**（batch=8，关闭早停，跑满）

## 2. 结果摘要

| 设置 | 数据 | mAP50-95 | 说明 |
|------|------|----------|------|
| 论文 YOLOv10-N | COCO val2017 全量 | **38.5%** | 论文报告值 |
| A 官方 `.pt` | coco128 | **45.71%** | 未训练 |
| B 100-epoch 微调 | coco128 | **75.73%** | 最佳约第 83 epoch |

## 3. 与原论文的差距 / 注意点

1. coco128 ≠ 全量 COCO val2017  
2. 本机 CPU ≠ 论文 GPU 延迟协议  
3. Path B 在同一小集上训练与验证，mAP 会偏高（过拟合风险），**不能**据此声称超过论文 SOTA  
4. 目标是作业允许的「小型数据集 + 少/有限 epoch」可重跑复现

## 4. 重跑

```bash
pip install -r requirements.txt
set KMP_DUPLICATE_LIB_OK=TRUE
python src/01_official_pt_eval.py
python src/02_finetune_few_epochs.py
python src/03_summarize.py
```

本仓库为 **AT4 期末**提交版（含 `paper.pdf`）。  
AT3 原仓库（保留不动）：https://github.com/GareyLee2001/AT3-Part2-YOLOv10-reproduction
