# AT3② Result Comparison

## vs original paper

| Setting | Dataset | mAP50-95 | Note |
|---|---|---:|---|
| Paper YOLOv10-N | COCO val2017 full | 38.5% | Official paper number |
| A Official `.pt` (ours) | coco128.yaml | 45.71% | No train; device=cpu |
| B Fine-tune until convergence (ours) | coco128.yaml | 75.73% | ran=100, best≈83; patience=0; device=cpu |

## Interpretation

- Path A/B use **coco128** (~128 images), a common Ultralytics small subset — still not full COCO.
- Gap vs paper is expected: subset + CPU + simplified setting.
- Path B shows the assignment-allowed *small data + few epoch* fine-tune path is runnable.
