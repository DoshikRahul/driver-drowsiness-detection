"""
Run 3: learned classifier on facial-landmark features.

Uses the features saved by extract_landmark_features.py (EAR, MAR, head pose
and the 52 MediaPipe blendshape scores). Several classifiers are compared on
the validation split; the best one is evaluated once on the test split and
saved for the web app.

Output: runs/landmark/test_metrics.json, runs/landmark/model_comparison.json
        ml_model/models/landmark_classifier.joblib

Usage:  python training/train_landmark_classifier.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "runs" / "landmark"
MODEL_OUT = ROOT / "ml_model" / "models" / "landmark_classifier.joblib"
CLASSES = ["drowsy", "not_drowsy"]


def load(split):
    d = np.load(FEAT / f"features_{split}.npz")
    keep = d["face"]
    # target: 1 = drowsy (positive class), 0 = not drowsy
    return d["X"][keep], (d["y"][keep] == 0).astype(int), list(d["feature_names"])


def candidates():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=0),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                                 early_stopping=True, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                             early_stopping=True, max_iter=500, random_state=0)),
    }


def main():
    Xtr, ytr, names = load("train")
    Xva, yva, _ = load("valid")
    Xte, yte, _ = load("test")
    geom_idx = list(range(9))  # ear, ear_left, ear_right, mar, blink, jaw_open, pitch, yaw, roll

    comparison = {}
    fitted = {}
    for fs_name, idx in (("geometry_only", geom_idx), ("all_features", list(range(len(names))))):
        for name, clf in candidates().items():
            clf.fit(Xtr[:, idx], ytr)
            p = clf.predict_proba(Xva[:, idx])[:, 1]
            key = f"{name}/{fs_name}"
            comparison[key] = {"valid_accuracy": round(accuracy_score(yva, p >= 0.5), 4),
                               "valid_f1_drowsy": round(f1_score(yva, p >= 0.5), 4),
                               "valid_roc_auc": round(roc_auc_score(yva, p), 4)}
            fitted[key] = (clf, idx)
            print(f"{key:45s} {comparison[key]}")

    best_key = max(comparison, key=lambda k: (comparison[k]["valid_roc_auc"], comparison[k]["valid_accuracy"]))
    clf, idx = fitted[best_key]
    p = clf.predict_proba(Xte[:, idx])[:, 1]
    pred = (p >= 0.5).astype(int)
    # report in the dataset's label order (drowsy, not_drowsy)
    y_lbl = np.where(yte == 1, 0, 1)
    p_lbl = np.where(pred == 1, 0, 1)
    rep = classification_report(y_lbl, p_lbl, target_names=CLASSES, output_dict=True)
    metrics = {
        "model": f"Landmark classifier: {best_key}",
        "split": "test",
        "n_images": int(len(yte)),
        "accuracy": round(rep["accuracy"], 4),
        "drowsy_precision": round(rep["drowsy"]["precision"], 4),
        "drowsy_recall": round(rep["drowsy"]["recall"], 4),
        "drowsy_f1": round(rep["drowsy"]["f1-score"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(y_lbl, p_lbl).tolist()},
        "selected_on": "validation ROC-AUC",
    }
    (FEAT / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (FEAT / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": [names[i] for i in idx], "positive_class": "drowsy",
                 "selected": best_key, "test_accuracy": metrics["accuracy"]}, MODEL_OUT)
    print(f"\nBest on validation: {best_key}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
