# -*- coding: utf-8 -*-
"""
AT3② Path B: small-data fine-tune of YOLOv10n on coco128.
Protocol: coco128, imgsz=640, fixed 100 epochs (manageable on CPU).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "finetune"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov10n.pt"
DATA = "coco128.yaml"  # common small set (~128 images), more than coco8
IMGSZ = 640
MAX_EPOCHS = 100
PATIENCE = 0  # run all 100 epochs
BATCH = 8
DEVICE = "cpu"


def find_demo_images() -> Path | None:
    datasets_dir = Path(SETTINGS.get("datasets_dir", Path.home() / "datasets"))
    for p in [
        datasets_dir / "coco128" / "images" / "train2017",
        datasets_dir / "coco128" / "images" / "val2017",
        datasets_dir / "coco128" / "images" / "train",
        datasets_dir / "coco128" / "images" / "val",
    ]:
        if p.exists():
            return p
    return None


def read_train_curve(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {"epochs_ran": None, "best_epoch": None, "best_map50_95": None}
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    if not rows:
        return {"epochs_ran": 0, "best_epoch": None, "best_map50_95": None}
    map_key = next(
        (k for k in rows[0].keys() if "mAP50-95" in k or k.strip().endswith("mAP50-95(B)")),
        None,
    )
    epoch_key = next((k for k in rows[0].keys() if k.strip().lower() == "epoch"), list(rows[0].keys())[0])
    best_i = 0
    best_v = float("-inf")
    for i, row in enumerate(rows):
        if map_key is None:
            break
        try:
            v = float(row[map_key])
        except (TypeError, ValueError):
            continue
        if v >= best_v:
            best_v = v
            best_i = i
    try:
        best_epoch = int(float(rows[best_i][epoch_key]))
    except Exception:
        best_epoch = best_i + 1
    return {
        "epochs_ran": len(rows),
        "best_epoch": best_epoch,
        "best_map50_95": None if map_key is None or best_v == float("-inf") else best_v,
        "map_column": map_key,
    }


def main() -> None:
    print(
        f"[B] Fine-tune {MODEL_NAME} on {DATA} for {MAX_EPOCHS} epochs "
        f"(batch={BATCH}, patience={PATIENCE})"
    )
    model = YOLO(MODEL_NAME)

    t0 = time.perf_counter()
    train_results = model.train(
        data=DATA,
        epochs=MAX_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(OUT),
        name="train",
        exist_ok=True,
        patience=PATIENCE,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    train_sec = time.perf_counter() - t0

    best = OUT / "train" / "weights" / "best.pt"
    last = OUT / "train" / "weights" / "last.pt"
    ckpt = best if best.exists() else last
    curve = read_train_curve(OUT / "train" / "results.csv")
    print(f"[B] Load fine-tuned checkpoint: {ckpt}")
    print(f"[B] Curve: epochs_ran={curve['epochs_ran']}, best_epoch={curve['best_epoch']}")
    ft = YOLO(str(ckpt))

    t1 = time.perf_counter()
    metrics = ft.val(
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
    val_sec = time.perf_counter() - t1

    img_dir = find_demo_images()
    if img_dir is not None:
        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:8]
        ft.predict(
            source=[str(p) for p in imgs] if imgs else str(img_dir),
            imgsz=IMGSZ,
            device=DEVICE,
            save=True,
            project=str(FIG),
            name="finetune_predict",
            exist_ok=True,
        )

    box = metrics.box
    summary = {
        "path": "B_finetune_coco128",
        "base_model": MODEL_NAME,
        "finetuned_checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.exists() else None,
        "data": DATA,
        "imgsz": IMGSZ,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epochs_ran": curve["epochs_ran"],
        "best_epoch": curve["best_epoch"],
        "best_map50_95_during_train": curve["best_map50_95"],
        "batch": BATCH,
        "device": DEVICE,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(getattr(box, "map75", float("nan"))),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "train_seconds": round(train_sec, 3),
        "val_seconds": round(val_sec, 3),
        "speed_ms_per_image": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "note": (
            f"Fixed {MAX_EPOCHS}-epoch fine-tune on coco128 "
            "(common Ultralytics small COCO subset; more than coco8, still not full COCO)."
        ),
        "train_save_dir": str(getattr(train_results, "save_dir", OUT / "train")),
    }
    out_json = OUT / "metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[B] Saved: {out_json}")


if __name__ == "__main__":
    main()
