# -*- coding: utf-8 -*-
"""
AT3② Path A: official YOLOv10n .pt -> fixed-subset evaluation + inference demo.
Protocol: coco128 (128 images; common Ultralytics demo set), imgsz=640, no training.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "official_pt"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov10n.pt"
DATA = "coco128.yaml"  # more commonly used than coco8; still small
IMGSZ = 640
DEVICE = "cpu"


def find_demo_images() -> Path | None:
    datasets_dir = Path(SETTINGS.get("datasets_dir", Path.home() / "datasets"))
    candidates = [
        datasets_dir / "coco128" / "images" / "train2017",
        datasets_dir / "coco128" / "images" / "val2017",
        datasets_dir / "coco128" / "images" / "train",
        datasets_dir / "coco128" / "images" / "val",
        ROOT / "datasets" / "coco128" / "images" / "train2017",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    print(f"[A] Load official weights: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"[A] Validate on fixed subset: {DATA}")
    t0 = time.perf_counter()
    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        device=DEVICE,
        split="val",
        project=str(OUT),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    val_sec = time.perf_counter() - t0

    img_dir = find_demo_images()
    if img_dir is not None:
        # only a few images for screenshot deliverable
        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:8]
        print(f"[A] Inference demo: {len(imgs)} images from {img_dir}")
        model.predict(
            source=[str(p) for p in imgs] if imgs else str(img_dir),
            imgsz=IMGSZ,
            device=DEVICE,
            save=True,
            project=str(FIG),
            name="official_pt_predict",
            exist_ok=True,
        )
    else:
        print("[A] WARN: coco128 images folder not found for predict demo")

    box = metrics.box
    summary = {
        "path": "A_official_pt",
        "model": MODEL_NAME,
        "data": DATA,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(getattr(box, "map75", float("nan"))),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "val_seconds": round(val_sec, 3),
        "speed_ms_per_image": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "note": "Official checkpoint only. No fine-tuning. Fixed subset = coco128 (common small COCO subset).",
    }
    out_json = OUT / "metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[A] Saved: {out_json}")


if __name__ == "__main__":
    main()
