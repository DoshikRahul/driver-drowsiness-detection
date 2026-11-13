"""
Collect the test metrics of every run into runs/summary.json, runs/summary.md
and a comparison chart runs/model_comparison.png.

Usage:  python training/summarize_runs.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SOURCES = [
    ("Run 1: Rule-based landmarks (EAR/MAR/pitch thresholds)", RUNS / "rule_based" / "test_metrics.json"),
    ("Run 2: CNN, YOLOv8n-cls (fine-tuned)", RUNS / "classify" / "drowsiness_yolov8n_cls" / "test_metrics.json"),
    ("Run 3: Landmark features + gradient boosting", RUNS / "landmark" / "test_metrics.json"),
    ("Run 4: Ensemble (CNN + landmark classifier)", RUNS / "ensemble" / "test_metrics.json"),
]
COLS = ["accuracy", "drowsy_precision", "drowsy_recall", "drowsy_f1", "macro_f1", "roc_auc"]


def main():
    rows = []
    for label, path in SOURCES:
        if path.exists():
            m = json.loads(path.read_text())
            rows.append({"run": label, "model": m.get("model"), "n_test": m.get("n_images"),
                         **{c: m.get(c) for c in COLS}, "confusion_matrix": m.get("confusion_matrix")})
    dataset = json.loads((RUNS / "dataset_report.json").read_text())
    (RUNS / "summary.json").write_text(json.dumps({"dataset": dataset, "results": rows}, indent=2))

    fmt = lambda v: "–" if v is None else f"{100 * v:.1f}%" if isinstance(v, float) and v <= 1 else str(v)
    lines = ["# Drowsiness detection: test-set results", "",
             f"Dataset: {dataset['source']} (face images only, leakage-free group split)", "",
             "| Run | Accuracy | Drowsy precision | Drowsy recall | Drowsy F1 | Macro F1 | ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join([r["run"]] + [fmt(r[c]) if c != "roc_auc" or r[c] is None else f"{r[c]:.3f}"
                                                     for c in COLS]) + " |")
    (RUNS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["accuracy", "drowsy_recall", "drowsy_precision", "macro_f1"]
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(rows))
    colors = ["#9aa5b1", "#2f6f9f", "#e0a33a", "#3c9a5f"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rows):
        vals = [100 * (r[m] or 0) for m in metrics]
        bars = ax.bar(x + (i - (len(rows) - 1) / 2) * w, vals, w, label=r["run"].split(":")[0] + ": " +
                      r["run"].split(": ", 1)[1].split(" (")[0], color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Accuracy", "Drowsy recall", "Drowsy precision", "Macro F1"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Driver drowsiness: test-set comparison")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(RUNS / "model_comparison.png", dpi=150)
    print(f"Chart: {RUNS / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
