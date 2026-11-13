"""
Extract facial-landmark features for every dataset image and evaluate the
rule-based detector (Run 1).

For each image the MediaPipe-based DrowsinessDetector (ml_model/) is run and we
store: geometric measures (EAR, MAR, head pose), the 52 blendshape scores and
the rule-based state. Results go to runs/landmark/features_<split>.npz and
are reused by train_landmark_classifier.py.

Usage:  python training/extract_landmark_features.py [--workers 6]
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "driver_drowsiness"
OUT = ROOT / "runs" / "landmark"
CLASSES = ["drowsy", "not_drowsy"]
SPLITS = ["train", "valid", "test"]
sys.path.insert(0, str(ROOT / "ml_model"))

GEOM = ["ear", "ear_left", "ear_right", "mar", "blink", "jaw_open", "pitch", "yaw", "roll"]
_det = None


def _init():
    global _det
    import cv2  # noqa: F401
    from drowsiness_detector import DrowsinessDetector
    _det = DrowsinessDetector()


def _work(path):
    import cv2
    img = cv2.imread(path)
    t0 = time.perf_counter()
    r = _det.analyze_bgr(img, return_blendshapes=True)
    ms = 1000 * (time.perf_counter() - t0)
    if not r.get("face_detected"):
        return path, False, None, None, 4, ms
    geom = [r["metrics"][k] for k in GEOM]
    blend = r["blendshapes"]
    return path, True, geom, blend, r["state"], ms


def extract(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    blend_names = None
    with Pool(workers, initializer=_init) as pool:
        for split in SPLITS:
            files, labels = [], []
            for ci, cls in enumerate(CLASSES):
                fs = sorted(str(p) for p in (DATA / split / cls).glob("*.jpg"))
                files += fs
                labels += [ci] * len(fs)
            t0 = time.time()
            res = pool.map(_work, files, chunksize=32)
            face = np.array([r[1] for r in res])
            states = np.array([r[4] for r in res])
            ms = np.array([r[5] for r in res])
            if blend_names is None:
                blend_names = sorted(next(r[3] for r in res if r[1]).keys())
            X = np.full((len(res), len(GEOM) + len(blend_names)), np.nan, dtype=np.float32)
            for i, r in enumerate(res):
                if r[1]:
                    X[i, :len(GEOM)] = r[2]
                    X[i, len(GEOM):] = [r[3].get(n, 0.0) for n in blend_names]
            np.savez_compressed(OUT / f"features_{split}.npz", X=X, y=np.array(labels), face=face,
                                state=states, ms=ms, files=np.array([str(Path(f).relative_to(ROOT)) for f in files]),
                                feature_names=np.array(GEOM + blend_names))
            print(f"{split}: {len(files)} images, face found {face.mean():.1%}, "
                  f"{np.median(ms):.1f} ms/img median, {time.time() - t0:.0f}s")


def evaluate_rule_based():
    """Run 1: the hand-tuned EAR/MAR/pitch rules, scored on the test split."""
    from sklearn.metrics import classification_report, confusion_matrix

    d = np.load(OUT / "features_test.npz")
    y, state, face = d["y"], d["state"], d["face"]
    # rule-based: Eyes closed / Yawning / Head nodding -> drowsy (0); Alert or no face -> not_drowsy (1)
    pred = np.where(np.isin(state, [1, 2, 3]), 0, 1)
    rep = classification_report(y, pred, target_names=CLASSES, output_dict=True)
    rep_face = classification_report(y[face], pred[face], target_names=CLASSES, output_dict=True)
    names = {0: "Alert", 1: "Eyes closed", 2: "Yawning", 3: "Head nodding", 4: "No face"}
    metrics = {
        "model": "Rule-based landmarks (EAR < 0.21 / blink > 0.5, MAR > 0.6 / jawOpen > 0.5, pitch > 20°)",
        "split": "test",
        "n_images": int(len(y)),
        "face_detection_rate": round(float(face.mean()), 4),
        "accuracy": round(rep["accuracy"], 4),
        "drowsy_precision": round(rep["drowsy"]["precision"], 4),
        "drowsy_recall": round(rep["drowsy"]["recall"], 4),
        "drowsy_f1": round(rep["drowsy"]["f1-score"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "accuracy_on_detected_faces": round(rep_face["accuracy"], 4),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(y, pred).tolist()},
        "predicted_state_counts_by_true_class": {
            CLASSES[c]: {names[s]: int(((y == c) & (state == s)).sum()) for s in names} for c in (0, 1)},
        "ms_per_image_cpu": round(float(np.median(d["ms"])), 2),
    }
    rb = ROOT / "runs" / "rule_based"
    rb.mkdir(parents=True, exist_ok=True)
    (rb / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    if not a.skip_extract:
        extract(a.workers)
    evaluate_rule_based()
