# AT4 Final Report — YOLOv10 Reproduction & Integration

**Course:** DDS001 数据科学前沿分析研究  
**Student:** 李嘉烯　`D26092110161`  
**Paper:** Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024  

This repository is the **AT4 end-of-term** submission (paper PDF + reproducible code).  
It continues the AT3 Part2 technical reproduction; the original AT3 repo is kept separately:
https://github.com/GareyLee2001/AT3-Part2-YOLOv10-reproduction

## Repository layout (AT4)

```
├── README.md
├── requirements.txt
├── paper.pdf              # 8–10 page final report
├── REPRODUCTION_NOTE.md
├── src/                   # code
│   ├── 01_official_pt_eval.py
│   ├── 02_finetune_few_epochs.py
│   └── 03_summarize.py
└── results/               # figures & metrics
```

## What the code does

| Path | What | Command |
|------|------|---------|
| **A** | Official `.pt` → eval + inference | `python src/01_official_pt_eval.py` |
| **B** | Small-data fine-tune (100 epochs) | `python src/02_finetune_few_epochs.py` |
| Summary | Merge metrics + paper comparison | `python src/03_summarize.py` |

### Fixed protocol

- Model: `yolov10n.pt`
- Dataset: **coco128**
- Image size: 640
- Path B: **100 epochs**, batch=8
- Device: **CPU**

## Setup

```bash
pip install -r requirements.txt
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Reproduce

```bash
python src/01_official_pt_eval.py
python src/02_finetune_few_epochs.py
python src/03_summarize.py
```

## Latest results (coco128)

| Setting | mAP50-95 |
|---------|---------:|
| Paper YOLOv10-N (full COCO val) | 38.5% |
| A Official `.pt` | 45.71% |
| B 100-epoch fine-tune | 75.73% |

> coco128 ≪ full COCO；Path B 在同一小集上训/验，数字偏乐观，不能直接声称打赢论文。详见 `REPRODUCTION_NOTE.md`。

## License / attribution

YOLOv10 via Ultralytics. Paper: Wang et al., NeurIPS 2024.
