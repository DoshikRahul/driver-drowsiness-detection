import logging
import os
import uuid
from io import BytesIO

from django.contrib import messages
from django.core.files.base import ContentFile
from django.db.models import Avg, Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .forms import ImageUploadForm
from .ml_service import service
from .models import DrowsinessAnalysis

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 5 * 1024 * 1024


def home(request):
    return render(request, "detector/home.html")


def upload_image(request):
    """Upload a driver image and analyse it for drowsiness."""
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            analysis = None
            try:
                uploaded = form.cleaned_data["image"]
                # Save first so the detector can read the stored file
                analysis = DrowsinessAnalysis(state=DrowsinessAnalysis.NO_FACE, uploaded_image=uploaded)
                analysis.save()

                result, annotated = service.analyze_image(analysis.uploaded_image.path)
                if not result.get("success"):
                    analysis.uploaded_image.delete(save=False)
                    analysis.delete()
                    messages.error(request, f"Analysis failed: {result.get('error', 'Unknown error')}")
                    return render(request, "detector/upload.html", {"form": form})

                filled = DrowsinessAnalysis.from_result(result)
                for field in ("state", "confidence", "drowsiness_score", "is_drowsy", "face_detected",
                              "ear", "mar", "head_pitch", "model_probability", "metrics_json", "scores_json"):
                    setattr(analysis, field, getattr(filled, field))
                if annotated:
                    name = os.path.splitext(os.path.basename(analysis.uploaded_image.name))[0]
                    analysis.annotated_image.save(f"{name}_annotated.jpg", ContentFile(annotated), save=False)
                analysis.save()
                return redirect("analysis_result", analysis_id=analysis.id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error processing upload")
                if analysis is not None and analysis.pk:
                    analysis.delete()
                messages.error(request, f"Error processing image: {exc}")
        else:
            messages.error(request, "Please select a valid image file.")
    else:
        form = ImageUploadForm()

    return render(request, "detector/upload.html", {"form": form})


def analysis_result(request, analysis_id):
    analysis = get_object_or_404(DrowsinessAnalysis, id=analysis_id)
    scores = analysis.scores
    chart_data = {
        "labels": ["Eyes closed", "Yawning", "Head nodding", "Alert"],
        "data": [scores.get(k, 0.0) for k in ("eyes_closed", "yawning", "head_nodding", "alert")],
    }
    return render(request, "detector/result.html", {
        "analysis": analysis,
        "metrics": analysis.metrics,
        "chart_data": chart_data,
    })


def history(request):
    qs = DrowsinessAnalysis.objects.all()
    stats = qs.aggregate(
        total=Count("id"),
        drowsy=Count("id", filter=Q(is_drowsy=True)),
        alert=Count("id", filter=Q(is_drowsy=False, face_detected=True)),
        avg_score=Avg("drowsiness_score"),
    )
    with_face = stats["drowsy"] + stats["alert"]
    stats["alert_rate"] = round(100 * stats["alert"] / with_face) if with_face else 0
    stats["avg_score"] = round((stats["avg_score"] or 0) * 100)
    return render(request, "detector/history.html", {"analyses": qs[:24], "stats": stats})


def download_report(request, analysis_id):
    """Download the analysis as a PDF report."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    analysis = get_object_or_404(DrowsinessAnalysis, id=analysis_id)
    m = analysis.metrics

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    p.setFont("Helvetica-Bold", 18)
    p.drawString(72, height - 72, "Driver Drowsiness Detection Report")
    p.setFont("Helvetica", 11)
    lines = [
        f"Analysis ID: #{analysis.id}",
        f"Date: {analysis.created_at:%Y-%m-%d %H:%M:%S}",
        f"Source: {analysis.get_source_display()}",
        "",
        f"Driver state: {analysis.state_name}",
        f"Overall status: {analysis.status_label}",
        f"Risk level: {analysis.risk_level}",
        f"Confidence: {analysis.confidence:.0%}",
        f"Drowsiness score: {analysis.drowsiness_percent}%",
    ]
    if m:
        lines += [
            "",
            f"Eye Aspect Ratio (EAR): {m.get('ear', 0):.3f}  (closed below 0.21)",
            f"Mouth Aspect Ratio (MAR): {m.get('mar', 0):.3f}  (yawn above 0.60)",
            f"Head pitch: {m.get('pitch', 0):.1f} deg  (nodding above 20)",
            f"Eye blink score: {m.get('blink', 0):.2f}   Jaw open score: {m.get('jaw_open', 0):.2f}",
        ]
    y = height - 110
    for line in lines:
        p.drawString(72, y, line)
        y -= 18

    img_field = analysis.annotated_image or analysis.uploaded_image
    try:
        img = ImageReader(img_field.path)
        iw, ih = img.getSize()
        scale = min((width - 144) / iw, (y - 72) / ih, 1.0)
        p.drawImage(img, 72, y - 20 - ih * scale, iw * scale, ih * scale)
    except Exception:  # noqa: BLE001
        logger.warning("Could not embed image in report", exc_info=True)

    p.showPage()
    p.save()
    buffer.seek(0)
    response = HttpResponse(buffer, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="drowsiness_report_{analysis_id}.pdf"'
    return response


def live_monitor(request):
    return render(request, "detector/live.html")


@require_POST
def api_analyze_frame(request):
    """Analyse one webcam frame (multipart field 'frame').

    With save=1 the frame and result are stored in history (used when the
    live monitor raises an alarm).
    """
    frame = request.FILES.get("frame")
    if frame is None:
        return JsonResponse({"success": False, "error": "No frame provided"}, status=400)
    if frame.size > MAX_FRAME_BYTES:
        return JsonResponse({"success": False, "error": "Frame too large"}, status=413)

    data = frame.read()
    save = request.POST.get("save") == "1"
    result, annotated = service.analyze_bytes(data, annotate=save)
    if not result.get("success"):
        return JsonResponse(result, status=500)

    if save:
        analysis = DrowsinessAnalysis.from_result(result, source="live", note=request.POST.get("note", "")[:255])
        name = f"live_{uuid.uuid4().hex[:10]}"
        analysis.uploaded_image.save(f"{name}.jpg", ContentFile(data), save=False)
        if annotated:
            analysis.annotated_image.save(f"{name}_annotated.jpg", ContentFile(annotated), save=False)
        analysis.save()
        result["saved_id"] = analysis.id
    return JsonResponse(result)


@csrf_exempt
@require_POST
def api_predict(request):
    """Stateless JSON API: POST an 'image' file, get drowsiness metrics back."""
    image = request.FILES.get("image")
    if image is None:
        return JsonResponse({"success": False, "error": "No image provided (field 'image')"}, status=400)
    if image.size > 10 * 1024 * 1024:
        return JsonResponse({"success": False, "error": "Image too large"}, status=413)
    result, _ = service.analyze_bytes(image.read())
    return JsonResponse(result, status=200 if result.get("success") else 500)


def model_info(request):
    from drowsiness_detector import (
        BLINK_CLOSED, EAR_CLOSED, JAW_YAWN, MAR_YAWN, PITCH_NOD_DEG,
    )

    info = {
        "status": service.status,
        "landmark_model": "MediaPipe Face Landmarker (478 landmarks + 52 blendshapes)",
        "thresholds": [
            {"name": "Eye Aspect Ratio (EAR)", "value": f"< {EAR_CLOSED}", "meaning": "Eyes closed"},
            {"name": "Eye blink blendshape", "value": f"> {BLINK_CLOSED}", "meaning": "Eyes closed"},
            {"name": "Mouth Aspect Ratio (MAR)", "value": f"> {MAR_YAWN}", "meaning": "Yawning"},
            {"name": "Jaw open blendshape", "value": f"> {JAW_YAWN}", "meaning": "Yawning"},
            {"name": "Head pitch", "value": f"> {PITCH_NOD_DEG:.0f}°", "meaning": "Head nodding"},
        ],
        "states": [
            {"name": "Alert", "risk": "LOW", "color": "success", "icon": "eye"},
            {"name": "Yawning", "risk": "MEDIUM", "color": "warning", "icon": "face-tired"},
            {"name": "Eyes closed", "risk": "HIGH", "color": "danger", "icon": "eye-slash"},
            {"name": "Head nodding", "risk": "HIGH", "color": "danger", "icon": "bed"},
            {"name": "Fatigue signs", "risk": "MEDIUM", "color": "warning", "icon": "battery-quarter"},
        ],
        "learned": service.learned_status,
        "evaluation": _load_evaluation(),
    }
    return render(request, "detector/model_info.html", {"info": info})


def _load_evaluation():
    """Test-set results of the training runs (runs/summary.json), if present."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "runs" / "summary.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    ds = data.get("dataset", {})
    counts = ds.get("new_split_counts", {})
    return {
        "results": data.get("results", []),
        "source": ds.get("source"),
        "train": sum(counts.get("train", {}).values()),
        "valid": sum(counts.get("valid", {}).values()),
        "test": sum(counts.get("test", {}).values()),
    }
