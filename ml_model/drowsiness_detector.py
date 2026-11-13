"""
Driver drowsiness detection using facial landmarks.

Uses MediaPipe Face Landmarker (478 3-D landmarks + 52 blendshapes) and
classic drowsiness indicators:

* EAR  - Eye Aspect Ratio (Soukupova & Cech, 2016). Drops towards 0 when
         the eyes close.
* MAR  - Mouth Aspect Ratio. Rises when the mouth opens wide (yawning).
* Head pitch - forward head tilt, a sign of nodding off.

Blendshape scores (eyeBlinkLeft/Right, jawOpen) are fused with the
geometric ratios to make the result robust to face size and camera angle.

A single image gives a per-frame state. Temporal logic (how long the eyes
stay closed, PERCLOS, yawn frequency) is applied by the live webcam view.
"""

import math
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Face Mesh topology)
# Order per eye: p1 (outer corner), p2, p3 (upper lid), p4 (inner corner),
#                p5, p6 (lower lid)  ->  EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)
# ---------------------------------------------------------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth: corners (61, 291) and three vertical pairs on the inner lips
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = [(81, 178), (13, 14), (311, 402)]

# ---------------------------------------------------------------------------
# Driver states
# ---------------------------------------------------------------------------
ALERT = 0
EYES_CLOSED = 1
YAWNING = 2
HEAD_NODDING = 3
NO_FACE = 4

STATE_NAMES = {
    ALERT: "Alert",
    EYES_CLOSED: "Eyes closed",
    YAWNING: "Yawning",
    HEAD_NODDING: "Head nodding",
    NO_FACE: "No face detected",
}

DROWSY_STATES = {EYES_CLOSED, YAWNING, HEAD_NODDING}

# Default thresholds (tunable)
EAR_CLOSED = 0.21      # below -> eyes considered closed
EAR_OPEN = 0.30        # at/above -> eyes fully open (used for scoring)
MAR_YAWN = 0.60        # above -> yawning
BLINK_CLOSED = 0.50    # blendshape eyeBlink score above -> closed
JAW_YAWN = 0.50        # blendshape jawOpen score above -> yawning
PITCH_NOD_DEG = 20.0   # head pitched down more than this -> nodding


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def ensure_model(model_path=DEFAULT_MODEL_PATH):
    """Download the face landmarker model if it is not present."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class DrowsinessDetector:
    """Detect driver drowsiness indicators in a single image / video frame."""

    MOUTH_OUTLINE = [MOUTH_CORNERS[0]] + [a for a, _ in MOUTH_VERTICAL] + [MOUTH_CORNERS[1]] + [
        b for _, b in reversed(MOUTH_VERTICAL)
    ]

    def __init__(self, model_path=None, auto_download=True):
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        if auto_download:
            ensure_model(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Face landmarker model not found at {model_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.model_path = str(model_path)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        # The landmarker is not thread-safe; Django's dev server is threaded.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _eye_aspect_ratio(pts, idx):
        p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal == 0:
            return 0.0
        return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))

    @staticmethod
    def _mouth_aspect_ratio(pts):
        horizontal = np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
        if horizontal == 0:
            return 0.0
        vertical = sum(np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_VERTICAL)
        return float(vertical / (len(MOUTH_VERTICAL) * horizontal))

    @staticmethod
    def _head_pose(matrix):
        """Return (pitch, yaw, roll) in degrees from a 4x4 transform matrix.

        Positive pitch = head tilted down (chin towards chest).
        """
        r = np.asarray(matrix)[:3, :3]
        sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            yaw = math.degrees(math.atan2(-r[2, 0], sy))
            roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
            yaw = math.degrees(math.atan2(-r[2, 0], sy))
            roll = 0.0
        # Verified empirically: looking down gives a positive x-rotation.
        return pitch, yaw, roll

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #
    def analyze_bgr(self, image_bgr, return_blendshapes=False):
        """Analyse a BGR (OpenCV) image. Returns a result dict.

        With return_blendshapes=True the full 52-score blendshape dict is
        included (used for training / the learned classifier).
        """
        import mediapipe as mp

        if image_bgr is None or image_bgr.size == 0:
            return {"success": False, "error": "Empty image"}

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        with self._lock:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {
                "success": True,
                "face_detected": False,
                "state": NO_FACE,
                "state_name": STATE_NAMES[NO_FACE],
                "is_drowsy": False,
                "confidence": 0.0,
                "drowsiness_score": 0.0,
                "metrics": {},
                "scores": {"eyes_closed": 0.0, "yawning": 0.0, "head_nodding": 0.0, "alert": 0.0},
                "flags": {"eyes_closed": False, "yawning": False, "head_nodding": False},
                "outline": None,
                "message": "No face detected. Make sure the driver's face is clearly visible.",
                "image_size": (w, h),
                "landmarks": None,
            }

        lms = result.face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float64)

        ear_left = self._eye_aspect_ratio(pts, LEFT_EYE)
        ear_right = self._eye_aspect_ratio(pts, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        mar = self._mouth_aspect_ratio(pts)

        blend = {}
        if result.face_blendshapes:
            blend = {b.category_name: float(b.score) for b in result.face_blendshapes[0]}
        blink = (blend.get("eyeBlinkLeft", 0.0) + blend.get("eyeBlinkRight", 0.0)) / 2.0
        jaw_open = blend.get("jawOpen", 0.0)

        pitch = yaw = roll = 0.0
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = self._head_pose(result.facial_transformation_matrixes[0])

        # ---- Fuse indicators into 0..1 scores --------------------------------
        ear_score = _clamp01((EAR_OPEN - ear) / (EAR_OPEN - EAR_CLOSED + 0.06))
        eyes_closed_score = max(ear_score, blink) if blend else ear_score
        mar_score = _clamp01((mar - 0.2) / (MAR_YAWN - 0.2 + 0.2))
        yawn_score = max(mar_score, jaw_open) if blend else mar_score
        nod_score = _clamp01((pitch - 5.0) / (PITCH_NOD_DEG - 5.0 + 10.0))

        eyes_closed = ear < EAR_CLOSED or blink > BLINK_CLOSED
        yawning = mar > MAR_YAWN or jaw_open > JAW_YAWN
        nodding = pitch > PITCH_NOD_DEG

        # Priority: closed eyes are the most dangerous sign, then nodding, then yawning
        if eyes_closed:
            state, confidence = EYES_CLOSED, max(eyes_closed_score, 0.5)
        elif nodding:
            state, confidence = HEAD_NODDING, max(nod_score, 0.5)
        elif yawning:
            state, confidence = YAWNING, max(yawn_score, 0.5)
        else:
            state = ALERT
            confidence = 1.0 - max(eyes_closed_score, yawn_score, nod_score)

        drowsiness_score = _clamp01(0.6 * eyes_closed_score + 0.25 * yawn_score + 0.15 * nod_score)
        if state in DROWSY_STATES:
            drowsiness_score = max(drowsiness_score, 0.5)

        scores = {
            "eyes_closed": round(eyes_closed_score, 4),
            "yawning": round(yawn_score, 4),
            "head_nodding": round(nod_score, 4),
            "alert": round(1.0 - max(eyes_closed_score, yawn_score, nod_score), 4),
        }

        result = {
            "success": True,
            "face_detected": True,
            "state": state,
            "state_name": STATE_NAMES[state],
            "is_drowsy": state in DROWSY_STATES,
            "confidence": round(float(_clamp01(confidence)), 4),
            "drowsiness_score": round(drowsiness_score, 4),
            "metrics": {
                "ear": round(ear, 4),
                "ear_left": round(ear_left, 4),
                "ear_right": round(ear_right, 4),
                "mar": round(mar, 4),
                "blink": round(blink, 4),
                "jaw_open": round(jaw_open, 4),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
            },
            "scores": scores,
            "flags": {"eyes_closed": bool(eyes_closed), "yawning": bool(yawning), "head_nodding": bool(nodding)},
            # Normalised (0..1) outlines for client-side overlays
            "outline": {
                "left_eye": [[round(lms[i].x, 4), round(lms[i].y, 4)] for i in LEFT_EYE],
                "right_eye": [[round(lms[i].x, 4), round(lms[i].y, 4)] for i in RIGHT_EYE],
                "mouth": [[round(lms[i].x, 4), round(lms[i].y, 4)] for i in self.MOUTH_OUTLINE],
            },
            "message": self._message(state),
            "image_size": (w, h),
            "landmarks": pts,
        }
        if return_blendshapes:
            result["blendshapes"] = blend
        return result

    def analyze_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            # OpenCV can't read some formats (e.g. some WEBP/HEIC); try Pillow
            try:
                from PIL import Image

                with Image.open(image_path) as im:
                    image = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": f"Could not read image: {exc}"}
        return self.analyze_bgr(image)

    @staticmethod
    def _message(state):
        return {
            ALERT: "Driver appears alert: eyes open, no yawning, head upright.",
            EYES_CLOSED: "DANGER: Driver's eyes are closed. Possible microsleep.",
            YAWNING: "WARNING: Driver is yawning, an early sign of fatigue.",
            HEAD_NODDING: "DANGER: Driver's head is tilting forward. Possible nodding off.",
            NO_FACE: "No face detected.",
        }[state]

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def annotate(self, image_bgr, result):
        """Draw eye / mouth landmarks and a status banner on a copy of the image."""
        img = image_bgr.copy()
        h, w = img.shape[:2]
        scale = max(w, h) / 800.0
        thickness = max(1, int(round(2 * scale)))

        state = result.get("state", NO_FACE)
        color = (0, 200, 0) if state == ALERT else (0, 165, 255) if state in (YAWNING, 5) else (0, 0, 255)
        if state == NO_FACE:
            color = (128, 128, 128)

        pts = result.get("landmarks")
        if pts is not None:
            for idx in (LEFT_EYE, RIGHT_EYE, self.MOUTH_OUTLINE):
                cv2.polylines(img, [np.array([pts[i] for i in idx], np.int32)], True, color, thickness)
            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        m = result.get("metrics") or {}
        label = result.get("state_name", "")
        if m:
            label += f"  EAR {m['ear']:.2f}  MAR {m['mar']:.2f}  Pitch {m['pitch']:.0f}"
        font_scale = 0.6 * scale + 0.2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (0, 0), (min(w, tw + 20), th + 20), color, -1)
        cv2.putText(img, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        return img


if __name__ == "__main__":
    import sys

    detector = DrowsinessDetector()
    for path in sys.argv[1:]:
        res = detector.analyze_image(path)
        if not res["success"]:
            print(f"{path}: error - {res['error']}")
            continue
        print(f"{path}: {res['state_name']} (conf {res['confidence']:.2f}, "
              f"drowsiness {res['drowsiness_score']:.2f}) {res['metrics']}")
        img = cv2.imread(path)
        if img is not None:
            out = os.path.splitext(path)[0] + "_annotated.jpg"
            cv2.imwrite(out, detector.annotate(img, res))
            print(f"  annotated -> {out}")
