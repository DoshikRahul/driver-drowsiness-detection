"""
Run 4: ensemble of the CNN (run 2) and the landmark classifier (run 3).

Uses exactly the runtime code path of the web app (ml_model/drowsiness_models.py,
ONNX model through OpenCV DNN). On the validation split it selects
  * the CNN input: whole image or a face crop (1.4x / 1.8x landmark box),
  * the CNN weight w in  p = w * p_cnn + (1 - w) * p_landmark,
then reports the chosen configuration once on the test split.

Output: runs/ensemble/test_metrics.json, runs/ensemble/validation_search.json
        ml_model/models/ensemble.json  (configuration loaded by the web app)

Usage:  python training/evaluate_ensemble.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml_model"))
from drowsiness_detector import DrowsinessDetector  # noqa: E402
from drowsiness_models import ENSEMBLE_PATH, LearnedModels, face_crop  # noqa: E402

OUT = ROOT / "runs" / "ensemble"
CLASSES = ["drowsy", "not_drowsy"]
MODES = ["full", "crop_1.4", "crop_1.8"]


def collect(split, det, models):
    rows = []
    for ci, cls in enumerate(CLASSES):
        for f in sorted((ROOT / "datasets" / "driver_drowsiness" / split / cls).glob("*.jpg")):
            img = cv2.imread(str(f))
            r = det.analyze_bgr(img, return_blendshapes=True)
            if not r.get("face_detected"):
                continue
            row = {"y": int(ci == 0), "p_lm": models.landmark_prob(r)}
            for m in MODES:
                src = img if m == "full" else face_crop(img, r["landmarks"], float(m.split("_")[1]))
                row[m] = models.cnn_prob(src)
            rows.append(row)
    return rows


def main():
    det = DrowsinessDetector()
    models = LearnedModels()
    assert models.cnn is not None and models.landmark is not None, "Train runs 2 and 3 first."
    val = collect("valid", det, models)
    test = collect("test", det, models)
    yv = np.array([r["y"] for r in val])
    yt = np.array([r["y"] for r in test])

    search = {}
    for m in MODES:
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            p = np.array([w * r[m] + (1 - w) * r["p_lm"] for r in val])
            search[f"{m}|w={w}"] = {"mode": m, "w": float(w), "valid_accuracy": round(accuracy_score(yv, p >= 0.5), 4),
                                    "valid_roc_auc": round(roc_auc_score(yv, p), 4)}
    best = max(search.values(), key=lambda s: (s["valid_roc_auc"], s["valid_accuracy"]))
    cnn_only = {m: round(roc_auc_score(yv, [r[m] for r in val]), 4) for m in MODES}
    print("Validation ROC-AUC, CNN alone by input mode:", cnn_only)
    print("Best ensemble on validation:", best)

    p = np.array([best["w"] * r[best["mode"]] + (1 - best["w"]) * r["p_lm"] for r in test])
    pred = (p >= 0.5).astype(int)
    y_lbl, p_lbl = np.where(yt == 1, 0, 1), np.where(pred == 1, 0, 1)
    rep = classification_report(y_lbl, p_lbl, target_names=CLASSES, output_dict=True)
    metrics = {
        "model": f"Ensemble: {best['w']:.1f} x CNN ({best['mode']}) + {1 - best['w']:.1f} x landmark classifier",
        "split": "test",
        "n_images": int(len(yt)),
        "accuracy": round(rep["accuracy"], 4),
        "drowsy_precision": round(rep["drowsy"]["precision"], 4),
        "drowsy_recall": round(rep["drowsy"]["recall"], 4),
        "drowsy_f1": round(rep["drowsy"]["f1-score"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "roc_auc": round(float(roc_auc_score(yt, p)), 4),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(y_lbl, p_lbl).tolist()},
        "test_roc_auc_components": {
            "cnn_" + m: round(float(roc_auc_score(yt, [r[m] for r in test])), 4) for m in MODES} |
            {"landmark": round(float(roc_auc_score(yt, [r["p_lm"] for r in test])), 4)},
        "test_accuracy_cnn_onnx_full": round(accuracy_score(yt, np.array([r["full"] for r in test]) >= 0.5), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_search.json").write_text(json.dumps({"cnn_only_valid_auc": cnn_only, "grid": search}, indent=2))
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    ENSEMBLE_PATH.write_text(json.dumps({"weight_cnn": best["w"], "cnn_input": best["mode"], "threshold": 0.5,
                                         "selected_on": "validation ROC-AUC",
                                         "test_accuracy": metrics["accuracy"]}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
