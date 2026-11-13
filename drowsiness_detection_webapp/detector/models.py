import json

from django.db import models
from django.utils import timezone


class DrowsinessAnalysis(models.Model):
    """A single drowsiness analysis of an uploaded image or a live-monitor alert frame."""

    ALERT = 0
    EYES_CLOSED = 1
    YAWNING = 2
    HEAD_NODDING = 3
    NO_FACE = 4
    FATIGUE_SIGNS = 5  # trained model says drowsy, no single hand-written rule fired

    STATE_CHOICES = [
        (ALERT, "Alert"),
        (EYES_CLOSED, "Eyes closed"),
        (YAWNING, "Yawning"),
        (HEAD_NODDING, "Head nodding"),
        (NO_FACE, "No face detected"),
        (FATIGUE_SIGNS, "Fatigue signs"),
    ]

    SOURCE_CHOICES = [
        ("upload", "Image upload"),
        ("live", "Live monitor"),
    ]

    uploaded_image = models.ImageField(upload_to="uploads/")
    annotated_image = models.ImageField(upload_to="annotated/", blank=True, null=True)
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default="upload")

    state = models.IntegerField(choices=STATE_CHOICES)
    confidence = models.FloatField(default=0.0)
    drowsiness_score = models.FloatField(default=0.0)
    is_drowsy = models.BooleanField(default=False)
    face_detected = models.BooleanField(default=True)

    ear = models.FloatField(null=True, blank=True, help_text="Eye Aspect Ratio")
    mar = models.FloatField(null=True, blank=True, help_text="Mouth Aspect Ratio")
    head_pitch = models.FloatField(null=True, blank=True, help_text="Head pitch in degrees (+ = down)")
    model_probability = models.FloatField(null=True, blank=True,
                                          help_text="P(drowsy) from the trained CNN + landmark ensemble")

    metrics_json = models.TextField(default="{}")
    scores_json = models.TextField(default="{}")
    note = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Drowsiness analyses"

    def __str__(self):
        return f"{self.state_name} ({self.confidence:.0%}) @ {self.created_at:%Y-%m-%d %H:%M}"

    # ---- JSON helpers -------------------------------------------------
    @property
    def metrics(self):
        try:
            return json.loads(self.metrics_json)
        except (TypeError, ValueError):
            return {}

    @property
    def scores(self):
        try:
            return json.loads(self.scores_json)
        except (TypeError, ValueError):
            return {}

    @classmethod
    def from_result(cls, result, **kwargs):
        m = dict(result.get("metrics") or {})
        learned = result.get("learned")
        if learned:
            m["learned"] = learned
        m["rule_state"] = result.get("rule_state")
        return cls(
            state=result["state"],
            confidence=result.get("confidence", 0.0),
            drowsiness_score=result.get("drowsiness_score", 0.0),
            is_drowsy=result.get("is_drowsy", False),
            face_detected=result.get("face_detected", False),
            ear=m.get("ear"),
            mar=m.get("mar"),
            head_pitch=m.get("pitch"),
            model_probability=learned["p_drowsy"] if learned else None,
            metrics_json=json.dumps(m),
            scores_json=json.dumps(result.get("scores") or {}),
            **kwargs,
        )

    # ---- Display helpers ----------------------------------------------
    @property
    def state_name(self):
        return self.get_state_display()

    @property
    def status_label(self):
        if not self.face_detected:
            return "Unknown"
        return "Drowsy" if self.is_drowsy else "Alert"

    @property
    def alert_class(self):
        """Bootstrap contextual colour."""
        return {
            self.ALERT: "success",
            self.YAWNING: "warning",
            self.FATIGUE_SIGNS: "warning",
            self.EYES_CLOSED: "danger",
            self.HEAD_NODDING: "danger",
        }.get(self.state, "secondary")

    @property
    def icon(self):
        return {
            self.ALERT: "eye",
            self.EYES_CLOSED: "eye-slash",
            self.YAWNING: "face-tired",
            self.HEAD_NODDING: "bed",
            self.FATIGUE_SIGNS: "battery-quarter",
        }.get(self.state, "user-slash")

    @property
    def risk_level(self):
        if self.state in (self.EYES_CLOSED, self.HEAD_NODDING):
            return "HIGH"
        if self.state in (self.YAWNING, self.FATIGUE_SIGNS):
            return "MEDIUM"
        if self.state == self.ALERT:
            return "LOW"
        return "UNKNOWN"

    @property
    def learned(self):
        return self.metrics.get("learned")

    @property
    def model_percent(self):
        return None if self.model_probability is None else round(self.model_probability * 100)

    @property
    def drowsiness_percent(self):
        return round(self.drowsiness_score * 100)
