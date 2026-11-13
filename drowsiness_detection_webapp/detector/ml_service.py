"""
Drowsiness detection service: wraps the ml_model/ code for Django.

Every image goes through
  1. DrowsinessDetector  - MediaPipe landmarks -> EAR / MAR / head pose and a
                           rule-based state (Alert, Eyes closed, Yawning, ...)
  2. LearnedModels       - CNN + landmark classifier trained on the driver
                           drowsiness dataset -> P(drowsy)

The learned ensemble gives the final drowsy / not-drowsy verdict (it is far
more accurate on the test set, see runs/summary.md); the rule-based state is
kept as the human-readable explanation and drives the live-monitor timers.
"""

import logging
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ML_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_model"
sys.path.append(str(ML_MODEL_DIR))

from drowsiness_detector import (  # noqa: E402
    ALERT,
    DROWSY_STATES,
    STATE_NAMES,
    DrowsinessDetector,
)

FATIGUE_SIGNS = 5  # learned model says drowsy although no single rule fired
STATE_NAMES = {**STATE_NAMES, FATIGUE_SIGNS: "Fatigue signs"}


class DrowsinessService:
    """Lazily-initialised singleton around the detector and learned models."""

    def __init__(self):
        self._detector = None
        self._models = None
        self._error = None
        self._init_lock = threading.Lock()

    def _init(self):
        if (self._detector is None and self._error is None):
            with self._init_lock:
                if self._detector is None and self._error is None:
                    try:
                        self._detector = DrowsinessDetector()
                        logger.info("Drowsiness detector initialised (%s)", self._detector.model_path)
                    except Exception as exc:  # noqa: BLE001
                        self._error = str(exc)
                        logger.exception("Failed to initialise drowsiness detector")
                        return
                    try:
                        from drowsiness_models import LearnedModels

                        self._models = LearnedModels()
                        logger.info("Learned models: %s", self._models.available)
                    except Exception:  # noqa: BLE001
                        logger.exception("Learned models unavailable; using rule-based detection only")

    @property
    def detector(self):
        self._init()
        return self._detector

    @property
    def models(self):
        self._init()
        return self._models

    @property
    def status(self):
        if self.detector is not None:
            return "Model loaded"
        return f"Model unavailable: {self._error}"

    @property
    def learned_status(self):
        m = self.models
        return m.available if m is not None else {"cnn": False, "landmark": False}

    # ------------------------------------------------------------------ #
    def _run(self, image, annotate):
        result = self.detector.analyze_bgr(image, return_blendshapes=True)
        result["rule_state"] = result["state"]
        result["rule_is_drowsy"] = result["is_drowsy"]
        result["learned"] = None

        if result.get("face_detected") and self.models is not None:
            learned = self.models.predict(image, result)
            if learned["p_drowsy"] is not None:
                result["learned"] = learned
                result["drowsiness_score"] = learned["p_drowsy"]
                result["is_drowsy"] = learned["is_drowsy"]
                if learned["is_drowsy"] and result["state"] == ALERT:
                    result["state"] = FATIGUE_SIGNS
                    result["message"] = "WARNING: The trained model sees signs of fatigue in the driver's face."
                elif not learned["is_drowsy"] and result["state"] in DROWSY_STATES:
                    result["state"] = ALERT
                    result["message"] = "Driver appears alert (the trained model overrides a borderline rule)."
                result["state_name"] = STATE_NAMES[result["state"]]
                p = learned["p_drowsy"]
                result["confidence"] = round(p if learned["is_drowsy"] else 1 - p, 4)

        annotated = None
        if annotate and result.get("success"):
            ok, buf = cv2.imencode(".jpg", self.detector.annotate(image, result), [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotated = buf.tobytes() if ok else None
        public = {k: v for k, v in result.items() if k not in ("landmarks", "blendshapes")}
        return public, annotated

    def analyze_image(self, image_path, annotate=True):
        """Analyse an image file. Returns (public_result, annotated_jpeg_bytes | None)."""
        if self.detector is None:
            return {"success": False, "error": self.status}, None
        image = cv2.imread(str(image_path))
        if image is None:
            try:
                from PIL import Image

                with Image.open(image_path) as im:
                    image = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": f"Invalid image file: {exc}"}, None
        return self._run(image, annotate)

    def analyze_bytes(self, data, annotate=False):
        """Analyse an encoded image (e.g. a JPEG webcam frame)."""
        if self.detector is None:
            return {"success": False, "error": self.status}, None
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"success": False, "error": "Could not decode frame"}, None
        return self._run(image, annotate)


service = DrowsinessService()

__all__ = ["service", "STATE_NAMES", "DROWSY_STATES", "FATIGUE_SIGNS"]
