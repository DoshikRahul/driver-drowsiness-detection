import shutil
import tempfile
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .ml_service import service
from .models import DrowsinessAnalysis

DATA = Path(__file__).parent / "test_data"
TEMP_MEDIA = tempfile.mkdtemp()


def upload(name):
    path = DATA / name
    return SimpleUploadedFile(path.name, path.read_bytes(), content_type="image/webp")


def blank_jpeg():
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 127, np.uint8))
    return SimpleUploadedFile("blank.jpg", buf.tobytes(), content_type="image/jpeg")


@override_settings(MEDIA_ROOT=TEMP_MEDIA)
class DrowsinessViewsTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA, ignore_errors=True)

    def test_pages_render(self):
        for name in ("home", "upload_image", "live_monitor", "history", "model_info"):
            self.assertEqual(self.client.get(reverse(name)).status_code, 200, name)

    def test_upload_creates_consistent_analysis(self):
        resp = self.client.post(reverse("upload_image"), {"image": upload("driver_alert.webp")})
        a = DrowsinessAnalysis.objects.get()
        self.assertRedirects(resp, reverse("analysis_result", args=[a.id]))
        self.assertTrue(a.face_detected)
        self.assertIsNotNone(a.ear)
        # verdict and state must agree
        drowsy_states = {a.EYES_CLOSED, a.YAWNING, a.HEAD_NODDING, a.FATIGUE_SIGNS}
        self.assertEqual(a.is_drowsy, a.state in drowsy_states)
        if service.models is not None and any(service.learned_status.values()):
            self.assertIsNotNone(a.model_probability)
            self.assertEqual(a.is_drowsy, a.model_probability >= a.learned["threshold"])
        self.assertTrue(a.annotated_image)
        self.assertEqual(self.client.get(reverse("analysis_result", args=[a.id])).status_code, 200)
        pdf = self.client.get(reverse("download_report", args=[a.id]))
        self.assertEqual(pdf["Content-Type"], "application/pdf")

    def test_upload_eyes_closed_driver(self):
        self.client.post(reverse("upload_image"), {"image": upload("driver_eyes_closed.webp")})
        a = DrowsinessAnalysis.objects.get()
        self.assertEqual(a.state, DrowsinessAnalysis.EYES_CLOSED)  # clear case: rules and models agree
        self.assertTrue(a.is_drowsy)
        self.assertLess(a.ear, 0.21)

    def test_api_predict_no_face(self):
        resp = self.client.post(reverse("api_predict"), {"image": blank_jpeg()})
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["face_detected"])

    def test_analyze_frame_and_save(self):
        resp = self.client.post(reverse("api_analyze_frame"), {"frame": upload("driver_alert.webp")})
        data = resp.json()
        self.assertTrue(data["face_detected"])
        self.assertIn("ear", data["metrics"])
        self.assertEqual(len(data["outline"]["left_eye"]), 6)
        self.assertEqual(DrowsinessAnalysis.objects.count(), 0)

        resp = self.client.post(reverse("api_analyze_frame"),
                                {"frame": upload("driver_eyes_closed.webp"), "save": "1", "note": "test alarm"})
        saved = DrowsinessAnalysis.objects.get(id=resp.json()["saved_id"])
        self.assertEqual(saved.source, "live")
        self.assertTrue(saved.is_drowsy)
        self.assertEqual(self.client.get(reverse("history")).status_code, 200)

    def test_model_info_shows_evaluation(self):
        resp = self.client.get(reverse("model_info"))
        self.assertContains(resp, "Trained models")

    def test_analyze_frame_requires_frame(self):
        self.assertEqual(self.client.post(reverse("api_analyze_frame")).status_code, 400)
