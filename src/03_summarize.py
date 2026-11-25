# -*- coding: utf-8 -*-
"""Merge Path A/B metrics into a comparison table for the report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "official_pt" / "metrics.json"
B = ROOT / "results" / "finetune" / "metrics.json"
OUT_MD = ROOT / "results" / "comparison.md"
OUT_JSON = ROOT / "results" / "comparison.json"

# YOLOv10 paper (NeurIPS 2024) reported numbers for reference (COCO val2017, official setting).
# Source: Wang et al., YOLOv10 — values are from the paper table for YOLOv10-N (approx.).
PAPER_REF = {
    "model": "YOLOv10-N (paper)",
    "dataset": "COCO val2017 (full)",
    "map50_95": 38.5,
    "latency_note": "Paper reports low end-to-end latency on modern GPU; not comparable 1:1 to this CPU/coco8 run.",
}


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    a = load(A)
    b = load(B)
    payload = {"paper_reference": PAPER_REF, "path_A": a, "path_B": b}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# AT3② Result Comparison",
        "",
        "## vs original paper",
        "",
        "| Setting | Dataset | mAP50-95 | Note |",
        "|---|---|---:|---|",
        f"| Paper YOLOv10-N | COCO val2017 full | {PAPER_REF['map50_95']:.1f}% | Official paper number |",
    ]
    if a:
        lines.append(
            f"| A Official `.pt` (ours) | {a.get('data')} | {100 * float(a.get('map50_95', float('nan'))):.2f}% | "
            f"No train; device={a.get('device')} |"
        )
    if b:
        ep = b.get("epochs_ran", b.get("epochs"))
        best_ep = b.get("best_epoch")
        lines.append(
            f"| B Fine-tune until convergence (ours) | {b.get('data')} | {100 * float(b.get('map50_95', float('nan'))):.2f}% | "
            f"ran={ep}, best≈{best_ep}; patience={b.get('patience')}; device={b.get('device')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Path A/B use **coco128** (~128 images), a common Ultralytics small subset — still not full COCO.",
        "- Gap vs paper is expected: subset + CPU + simplified setting.",
        "- Path B shows the assignment-allowed *small data + few epoch* fine-tune path is runnable.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
