"""
Learned drowsiness models trained on the driver drowsiness dataset (see training/).

* CNN       - YOLOv8n-cls fine-tuned on face images, exported to ONNX and run
              with OpenCV DNN (no PyTorch needed at runtime).
* Landmark  - gradient-boosting classifier over MediaPipe landmark features
              (EAR, MAR, head pose, 52 blendshapes).
* Ensemble  - weighted average of the two probabilities; weight and CNN input
              mode were chosen on the validation split (models/ensemble.json).

All probabilities are P(drowsy) in [0, 1].
"""

import json
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
CNN_PATH = MODELS_DIR / "drowsiness_cls.onnx"
CNN_META = MODELS_DIR / "drowsiness_cls.json"
LANDMARK_PATH = MODELS_DIR / "landmark_classifier.joblib"
ENSEMBLE_PATH = MODELS_DIR / "ensemble.json"

GEOM = ["ear", "ear_left", "ear_right", "mar", "blink", "jaw_open", "pitch", "yaw", "roll"]


def face_crop(image_bgr, landmarks, scale):
    """Square crop around the landmark bounding box, enlarged by `scale`."""
    h, w = image_bgr.shape[:2]
    x1, y1 = landmarks.min(axis=0)
    x2, y2 = landmarks.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = image_bgr[b:d, a:c]
    return crop if crop.size else image_bgr


class LearnedModels:
    def __init__(self):
        self.cnn = None
        self.cnn_imgsz = 224
        self.cnn_drowsy_index = 0
        self.landmark = None
        self.landmark_features = None
        self.config = {"weight_cnn": 0.5, "cnn_input": "full", "threshold": 0.5}
        self._lock = threading.Lock()

        if CNN_PATH.exists():
            self.cnn = cv2.dnn.readNetFromONNX(str(CNN_PATH))
            if CNN_META.exists():
                meta = json.loads(CNN_META.read_text())
                self.cnn_imgsz = meta.get("imgsz", 224)
                names = {int(k): v for k, v in meta.get("names", {}).items()}
                self.cnn_drowsy_index = next((k for k, v in names.items() if v == "drowsy"), 0)
        if LANDMARK_PATH.exists():
            import joblib  # noqa: WPS433 (optional dependency: scikit-learn)

            bundle = joblib.load(LANDMARK_PATH)
            self.landmark = bundle["model"]
            self.landmark_features = bundle["feature_names"]
        if ENSEMBLE_PATH.exists():
            self.config.update(json.loads(ENSEMBLE_PATH.read_text()))

    @property
    def available(self):
        return {"cnn": self.cnn is not None, "landmark": self.landmark is not None}

    # ------------------------------------------------------------------ #
    def cnn_prob(self, image_bgr):
        """P(drowsy) from the CNN for a BGR image (Ultralytics classify preprocessing)."""
        if self.cnn is None:
            return None
        s = self.cnn_imgsz
        h, w = image_bgr.shape[:2]
        r = s / min(h, w)  # resize shortest side, then centre-crop
        img = cv2.resize(image_bgr, (max(s, round(w * r)), max(s, round(h * r))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (s, s), swapRB=True, crop=False)
        with self._lock:
            self.cnn.setInput(blob)
            out = self.cnn.forward().reshape(-1)
        if out.min() < 0 or abs(out.sum() - 1) > 1e-3:  # logits -> softmax
            out = np.exp(out - out.max())
            out /= out.sum()
        return float(out[self.cnn_drowsy_index])

    def landmark_prob(self, result):
        """P(drowsy) from the landmark classifier, given a detector result with blendshapes."""
        if self.landmark is None or not result.get("face_detected") or "blendshapes" not in result:
            return None
        m, b = result["metrics"], result["blendshapes"]
        x = [m[k] if k in m else b.get(k, 0.0) for k in self.landmark_features]
        return float(self.landmark.predict_proba(np.array([x], dtype=np.float32))[0, 1])

    def predict(self, image_bgr, result):
        """Combine both models. `result` must come from analyze_bgr(..., return_blendshapes=True)."""
        p_cnn = None
        if self.cnn is not None:
            img = image_bgr
            mode = self.config.get("cnn_input", "full")
            if mode.startswith("crop") and result.get("landmarks") is not None:
                img = face_crop(image_bgr, result["landmarks"], float(mode.split("_")[1]))
            p_cnn = self.cnn_prob(img)
        p_lm = self.landmark_prob(result)

        w = self.config.get("weight_cnn", 0.5)
        if p_cnn is not None and p_lm is not None:
            p = w * p_cnn + (1 - w) * p_lm
        else:
            p = p_cnn if p_cnn is not None else p_lm
        thr = self.config.get("threshold", 0.5)
        return {
            "p_cnn": None if p_cnn is None else round(p_cnn, 4),
            "p_landmark": None if p_lm is None else round(p_lm, 4),
            "p_drowsy": None if p is None else round(p, 4),
            "is_drowsy": None if p is None else bool(p >= thr),
            "threshold": thr,
        }
