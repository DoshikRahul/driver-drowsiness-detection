"""
Generate README.docx and EXECUTION_GUIDE.docx for the drowsiness detection project.

Results (dataset counts, test metrics, charts) are read from runs/, so re-run this
after any training run:   python docs/make_docs.py
Requires: pip install -r docs/requirements.txt
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT          # where the .docx files are written
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "assets"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x59, 0x59, 0x59)


# --------------------------------------------------------------------------- helpers
def new_doc(title, subtitle):
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.2))
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    st = d.styles
    st["Normal"].font.name = "Calibri"
    st["Normal"].font.size = Pt(11)
    st["Normal"].paragraph_format.space_after = Pt(6)
    st["Normal"].paragraph_format.line_spacing = 1.15
    for name, size, before in (("Heading 1", 16, 18), ("Heading 2", 13, 12), ("Heading 3", 11.5, 10)):
        s = st[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = NAVY
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(6)
        s.paragraph_format.keep_with_next = True

    t = d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(24)
    r = t.add_run(title)
    r.bold = True
    r.font.size = Pt(24)
    r.font.color.rgb = NAVY
    s = d.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run(subtitle)
    r.font.size = Pt(12.5)
    r.font.color.rgb = GREY
    rule(d)

    # footer page numbers
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run(f"{title}  ·  Page ")
    fr.font.size = Pt(9)
    fr.font.color.rgb = GREY
    field(fp, "PAGE")
    return d


def field(paragraph, code):
    run = paragraph.add_run()
    run.font.size = Pt(9)
    for tag, text in (("begin", None), (None, code), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def rule(d):
    p = d.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "8"), ("space", "1"), ("color", "1F3A5F")):
        bottom.set(qn(f"w:{k}"), v)
    bdr.append(bottom)
    pPr.append(bdr)


def shade(el, hex_fill):
    props = el.get_or_add_tcPr() if hasattr(el, "get_or_add_tcPr") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    props.append(shd)


def para(d, text, bold_lead=None, italic=False):
    p = d.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    add_inline(p, text, italic)
    return p


def add_inline(p, text, italic=False):
    """Supports `code` spans inside text."""
    parts = text.split("`")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.italic = italic
        if i % 2 == 1:
            r.font.name = "Consolas"
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0xA3, 0x1F, 0x34)


def bullets(d, items, style="List Bullet"):
    for it in items:
        p = d.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(3)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            add_inline(p, it[1])
        else:
            add_inline(p, it)


def code(d, text):
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = d.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(4) if i == 0 else Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.left_indent = Cm(0.4)
        shade(p._p, "F2F4F7")
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


def table(d, header, rows, widths_cm):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(10)
        shade(c._tc, "1F3A5F")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            add_inline(p, str(val))
            for r in p.runs:
                if r.font.name != "Consolas":
                    r.font.size = Pt(10)
            if ri % 2 == 1:
                shade(cells[i]._tc, "F2F4F7")
    set_widths(t, widths_cm)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def set_widths(t, widths_cm):
    """Set grid + cell widths (LibreOffice reads the grid, Word reads cells)."""
    tbl = t._tbl
    tblPr = tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(int(sum(widths_cm) * 567)))
    for gc, w in zip(tbl.tblGrid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w * 567)))
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)


def image(d, path, width_cm, caption):
    if not Path(path).exists():
        return
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = d.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREY


def note(d, label, text, fill="FFF4E5"):
    t = d.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_widths(t, [16.6])
    c = t.rows[0].cells[0]
    shade(c._tc, fill)
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.add_run(label + " ").bold = True
    add_inline(p, text)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


# =========================================================================== README
def build_readme():
    d = new_doc("Driver Drowsiness Detection",
                "Using Computer Vision and Deep Learning: Project README")

    d.add_heading("1. Overview", level=1)
    para(d, "Driver Drowsiness Detection is a Django web application that watches a driver's face "
            "through a webcam (or analyses an uploaded photo) and warns when it sees signs of fatigue: "
            "eyes closing, yawning, or the head nodding forward. When the driver's eyes stay closed for "
            "too long, the system sounds an audible alarm and records a snapshot of the event.")
    para(d, "It combines two approaches. First, Google's MediaPipe Face Landmarker locates 478 points on the "
            "face, and well-established geometric measures (the Eye Aspect Ratio, Mouth Aspect Ratio and head "
            "pose) explain what the driver is doing. Second, two models trained on a public driver drowsiness "
            "dataset (a fine-tuned YOLOv8 CNN and a gradient-boosting classifier over the landmark features) "
            "give the final drowsy / not-drowsy verdict. No special hardware is needed: everything runs on a "
            "normal CPU with a standard webcam.")

    d.add_heading("2. Problem Statement", level=1)
    para(d, "Drowsy driving is a major cause of road accidents. A fatigued driver has slower reactions, "
            "poorer judgement and may fall into a microsleep: a lapse of a few seconds during which "
            "the vehicle is effectively uncontrolled. Drivers are often poor judges of their own "
            "tiredness. An automatic monitor that notices the physical signs of fatigue and alerts the "
            "driver in real time can prevent these crashes.")

    d.add_heading("3. Objectives", level=1)
    bullets(d, [
        "Detect closed eyes, yawning and head nodding from a standard camera image.",
        "Monitor a live webcam stream in real time and distinguish normal blinks from dangerous eye closure.",
        "Raise an audible and visual alarm when drowsiness is detected, and log the event.",
        "Track longer-term fatigue indicators (PERCLOS, yawn frequency) and warn the driver to take a break.",
        "Provide a simple web interface, analysis history, downloadable PDF reports and a JSON API.",
    ])

    d.add_heading("4. How It Works", level=1)
    d.add_heading("4.1 Processing pipeline", level=2)
    table(d, ["Step", "Stage", "What happens"], [
        ["1", "Capture", "The browser grabs a webcam frame (~7 per second), or the user uploads a photo."],
        ["2", "Face landmarks", "MediaPipe Face Landmarker returns 478 3-D landmarks, 52 expression blendshape scores and a head transformation matrix."],
        ["3", "Measurement", "EAR is computed for both eyes, MAR for the mouth, and head pitch from the matrix."],
        ["4", "Frame classification", "Thresholds give the frame state: Alert, Eyes closed, Yawning, Head nodding or No face."],
        ["4b", "Trained models", "The CNN (face image) and the landmark classifier each estimate P(drowsy). Their weighted average is the final verdict."],
        ["5", "Temporal analysis", "In live mode, the browser tracks how long each state lasts, and computes PERCLOS and yawn counts."],
        ["6", "Alert", "An alarm sounds when eyes are closed ≥ 1.5 s or the head nods ≥ 2 s. A snapshot is saved to History."],
    ], [1.3, 3.4, 11.9])

    d.add_heading("4.2 Fatigue signals", level=2)
    para(d,
               "Six landmarks around each eye (two corners, two on the upper lid, two on the lower lid) give "
               "`EAR = (|p2 − p6| + |p3 − p5|) / (2 × |p1 − p4|)`. An open eye measures about 0.25–0.35. "
               "The value drops towards 0 as the eyelids close. The two eyes are averaged and combined with "
               "MediaPipe's eyeBlink blendshape score for extra robustness.", "Eye Aspect Ratio (EAR). ")
    p = para(d, "", "Mouth Aspect Ratio (MAR). ")
    add_inline(p, "The average inner-lip opening divided by the mouth width. A wide opening indicates a yawn. "
                  "It's combined with the jawOpen blendshape score.")
    p = para(d, "", "Head pitch. ")
    add_inline(p, "The forward/backward tilt of the head, taken from the 3-D facial transformation "
                  "matrix. A large downward tilt (chin dropping towards the chest) indicates nodding off.")

    d.add_heading("4.3 Detection thresholds", level=2)
    table(d, ["Signal", "Threshold", "Interpretation"], [
        ["Eye Aspect Ratio (EAR)", "< 0.21 (or calibrated)", "Eyes closed"],
        ["eyeBlink blendshape", "> 0.50", "Eyes closed"],
        ["Mouth Aspect Ratio (MAR)", "> 0.60", "Yawning"],
        ["jawOpen blendshape", "> 0.50", "Yawning"],
        ["Head pitch", "> 20° downward", "Head nodding"],
    ], [6.0, 4.6, 6.0])

    d.add_heading("4.4 Driver states", level=2)
    table(d, ["State", "Risk", "Meaning"], [
        ["Alert", "LOW", "Eyes open, mouth closed, head upright."],
        ["Yawning", "MEDIUM", "An early sign of fatigue."],
        ["Eyes closed", "HIGH", "Possible microsleep if it lasts more than a blink."],
        ["Head nodding", "HIGH", "The driver may be falling asleep."],
        ["Fatigue signs", "MEDIUM", "No single rule fired, but the trained models recognise a tired face."],
        ["No face detected", "UNKNOWN", "The face is not visible (turned away, covered or out of frame)."],
    ], [3.8, 2.6, 10.2])

    d.add_heading("4.5 Live monitor logic", level=2)
    bullets(d, [
        ("Calibration: ", "for the first ~3 seconds the system learns the driver's normal open-eye EAR and "
                          "sets the personal closed-eye threshold to 75% of it (bounded to 0.15–0.27). If "
                          "calibration cannot finish within 6 s, the default 0.21 is used."),
        ("Drowsiness alarm: ", "eyes closed continuously for 1.5 s or more, or head nodding for 2 s or more. "
                               "A repeating beep sounds, the video border flashes red, and a snapshot is saved."),
        ("Blink filtering: ", "closures shorter than 0.5 s are counted as normal blinks and shown as blinks per minute."),
        ("PERCLOS: ", "the percentage of time the eyes were closed over the last 60 seconds. Above 15% shows a "
                      "\"take a break\" fatigue warning."),
        ("Yawn counter: ", "a mouth opening that lasts 1 second or more counts as a yawn. Three or more yawns "
                           "in 5 minutes shows a fatigue warning."),
        ("Face lost: ", "no face for 3 seconds shows a \"driver not visible\" warning."),
    ])

    dataset_section(d)

    d.add_heading("6. Features", level=1)
    bullets(d, [
        ("Live Monitor: ", "real-time webcam analysis with landmark overlay, alarm sound (mutable), live "
                           "metrics, an EAR graph and session statistics."),
        ("Image Upload: ", "analyse a single photo. You get an annotated image, the measurements, an indicator "
                           "chart and safety advice."),
        ("History: ", "all uploads and live-alarm snapshots with summary statistics."),
        ("PDF Report: ", "a downloadable report for any analysis, including the annotated image."),
        ("REST API: ", "`POST /api/predict/` returns the full analysis as JSON."),
        ("Command-line tool: ", "analyse images and write annotated copies without starting the web server."),
        ("Admin panel: ", "browse and filter all records at `/admin/`."),
    ])

    d.add_heading("7. Screenshots", level=1)
    image(d, SHOTS / "live_monitor.png", 16, "Figure 1: Live Monitor raising a drowsiness alarm after the eyes stayed closed for 1.5 s")
    image(d, SHOTS / "result_page.png", 13.5, "Figure 2: Result page for an uploaded image (eyes closed detected)")

    d.add_heading("8. Technology Stack", level=1)
    table(d, ["Layer", "Technology", "Purpose"], [
        ["Backend", "Python 3.10+, Django 4.2", "Web framework, routing, database, file storage"],
        ["Computer vision", "MediaPipe Face Landmarker", "478 face landmarks, blendshapes, head pose"],
        ["Image processing", "OpenCV, NumPy, Pillow", "Image decoding, geometry, annotation drawing"],
        ["Deep learning", "PyTorch, Ultralytics YOLOv8 (training); OpenCV DNN + ONNX (runtime)", "CNN drowsiness classifier"],
        ["Machine learning", "scikit-learn", "Gradient-boosting classifier on landmark features"],
        ["Reports", "ReportLab", "PDF report generation"],
        ["Database", "SQLite", "Stores analysis records"],
        ["Frontend", "Bootstrap 5, Chart.js, Font Awesome", "Responsive UI, charts, icons"],
        ["Browser APIs", "WebRTC getUserMedia, Web Audio, Canvas", "Webcam capture, alarm sound, overlays"],
    ], [3.4, 5.6, 7.6])

    d.add_heading("9. Project Structure", level=1)
    code(d, """
DRIVER DROWSINESS DETECTION/
├── README.md / README.docx / EXECUTION_GUIDE.docx
├── ml_model/
│   ├── drowsiness_detector.py        Core detector (EAR, MAR, head pose, annotation)
│   ├── drowsiness_models.py          Trained CNN + landmark classifier (inference)
│   └── models/
│       ├── face_landmarker.task      Pre-trained MediaPipe model (~3.7 MB)
│       ├── drowsiness_cls.onnx       Trained CNN (YOLOv8n-cls, exported)
│       ├── landmark_classifier.joblib  Trained gradient-boosting classifier
│       └── ensemble.json             Ensemble weight chosen on validation data
├── scripts/                          Bash scripts: setup, pipeline steps, run_all, status
├── docs/                             Document generator + screenshots
├── RUNS_LINEUP.md                    Pipeline steps, dependencies, timings
├── training/                         Scripts that reproduce every run
│   ├── prepare_dataset.py            Download, clean, re-split the dataset
│   ├── extract_landmark_features.py  Landmark features + Run 1 (rule-based)
│   ├── train_cnn.py                  Run 2: YOLOv8n-cls training
│   ├── train_landmark_classifier.py  Run 3: landmark classifier
│   ├── evaluate_ensemble.py          Run 4: ensemble
│   └── summarize_runs.py             Summary table + comparison chart
├── datasets/driver_drowsiness/       Prepared images: train / valid / test
├── runs/                             Metrics, curves, confusion matrices, logs
├── drowsiness_detection_webapp/      Django project
│   ├── manage.py
│   ├── requirements.txt
│   ├── drowsiness_detection_web/     Project settings & root URLs
│   └── detector/                     Main Django app
│       ├── models.py                 DrowsinessAnalysis database model
│       ├── views.py                  Pages, upload handling, APIs, PDF report
│       ├── ml_service.py             Thread-safe wrapper around the detector
│       ├── urls.py / forms.py / admin.py / tests.py
│       ├── templates/detector/       home, live, upload, result, history, model_info
│       └── test_data/                Sample images used by the automated tests
└── legacy_distracted_driver/         Archived earlier distracted-driver version (not used)
""")

    d.add_heading("10. Application Pages and Endpoints", level=1)
    table(d, ["URL", "Description"], [
        ["`/`", "Home page"],
        ["`/live/`", "Live webcam drowsiness monitor"],
        ["`/upload/`", "Upload and analyse a single image"],
        ["`/result/<id>/`", "Result of a single analysis"],
        ["`/history/`", "History of analyses and alarm snapshots"],
        ["`/download/<id>/`", "Download a PDF report"],
        ["`/model-info/`", "How It Works: model status, thresholds and logic"],
        ["`/api/predict/`", "JSON API: POST an image (field `image`)"],
        ["`/api/analyze-frame/`", "Internal API used by the live monitor"],
        ["`/admin/`", "Django admin panel"],
    ], [5.0, 11.6])

    d.add_heading("11. Limitations", level=1)
    bullets(d, [
        "Dark sunglasses hide the eyes, so eye closure cannot be measured through them.",
        "Strong side-profile views (more than ~45° head turn) reduce landmark accuracy.",
        "Looking down (for example at a phone) lowers the eyelids and can read as \"eyes closed\" in a single photo. "
        "The live monitor's duration rules reduce such false alarms.",
        "Very low light or strong backlight reduces detection quality. An infrared camera would be needed for night driving.",
        "The training data are mostly still images collected from the web, not in-car video. People can appear in "
        "more than one split even though identical photos cannot, so real-world accuracy may be lower than the "
        "test-set figures.",
        "Dataset bias: every in-car photo in the dataset is labelled drowsy, and the not-drowsy images are "
        "almost all close-up faces. The test set shares this bias, so it cannot reveal it. In practice the "
        "trained models can over-report drowsiness for alert drivers photographed inside a car, especially when "
        "the mouth is open (talking looks like yawning in a still photo). Collecting alert in-car images is the "
        "most important next step.",
        "Single images cannot capture how long the eyes stay closed. That is why the live monitor's alarm stays "
        "duration-based (eyes closed ≥ 1.5 s, nodding ≥ 2 s). The trained models only add a fatigue warning, "
        "and only after a sustained high probability.",
        "The webcam only works on `localhost` or over HTTPS (a browser security rule).",
    ])

    d.add_heading("12. Future Enhancements", level=1)
    bullets(d, [
        "Run detection fully in the browser (MediaPipe for JavaScript) for offline, low-latency use on a phone or dashboard device.",
        "Infrared camera support for night-time monitoring.",
        "Train a temporal deep-learning model (e.g. CNN + LSTM) on an in-car drowsiness video dataset with subject-wise splits.",
        "SMS or e-mail notification to a fleet manager or family member when repeated alarms occur.",
        "Integration with vehicle systems (seat vibration, hazard lights).",
    ])

    d.add_heading("13. Quick Start", level=1)
    para(d, "See EXECUTION_GUIDE.docx for full step-by-step instructions. In short:")
    code(d, """
python -m venv .venv
source .venv/bin/activate          (Windows: .venv\\Scripts\\activate)
pip install -r drowsiness_detection_webapp/requirements.txt
cd drowsiness_detection_webapp
python manage.py migrate
python manage.py runserver
""")
    para(d, "Then open http://127.0.0.1:8000/ in Chrome, Edge or Firefox.")

    d.save(OUT / "README.docx")


def dataset_section(d):
    import json as _json
    runs = ROOT / "runs"
    ds = _json.loads((runs / "dataset_report.json").read_text())
    summ = _json.loads((runs / "summary.json").read_text())["results"]
    nc = ds["new_split_counts"]
    tot = lambda sp: sum(nc[sp].values())

    d.add_heading("5. Dataset and Training", level=1)
    d.add_heading("5.1 Dataset", level=2)
    para(d, "Source: the public Driver Drowsiness dataset on Hugging Face "
            "(`n7i5x9/driver-drowsiness-dataset`). It contains 23,116 images (640×640) labelled drowsy or "
            "not_drowsy, merged from four public sources: closed eyes and yawning are labelled drowsy; open eyes "
            "and no yawn are labelled not_drowsy. The dataset does not state a licence, so check the original "
            "sources before any commercial use.")
    para(d, "Two problems were found and fixed before training:")
    bullets(d, [
        ("Train/test leakage. ", f"The published splits contain augmented copies of the same photo in train, "
         f"validation and test: {ds['published_split_leakage']['train&test']['pct_of_test_unique']:.0f}% of the "
         "distinct test images also appear in train. Scores on those splits would measure memorisation. The data "
         "were therefore re-split by original photo (all augmented copies stay together, 80/10/10, stratified by "
         f"class), and {sum(ds['dropped_cross_split_duplicates'].values())} remaining cross-split duplicates were "
         "removed using a perceptual hash. After this, no image is shared between splits."),
        ("Eye-only crops. ", f"{ds['face_filter']['images_without_face']:,} images are close-ups of a single eye "
         "rather than a face. This application analyses the whole face from a webcam, so images in which no face "
         "can be detected were excluded."),
    ])
    table(d, ["Split", "Drowsy", "Not drowsy", "Total"], [
        [sp.capitalize(), f"{nc[sp].get('drowsy', 0):,}", f"{nc[sp].get('not_drowsy', 0):,}", f"{tot(sp):,}"]
        for sp in ("train", "valid", "test")] + [["All", "", "", f"{sum(tot(sp) for sp in nc):,}"]],
        [4.0, 4.2, 4.2, 4.2])

    d.add_heading("5.2 Training runs", level=2)
    table(d, ["Run", "Method", "Details"], [
        ["1", "Rule-based landmarks (baseline)", "The hand-set thresholds of section 4.3, no training. Drowsy = Eyes closed, Yawning or Head nodding."],
        ["2", "CNN: YOLOv8n-cls", "ImageNet-pretrained YOLOv8 nano classifier fine-tuned on the face images (224×224, AdamW, cosine LR, up to 20 epochs, early stopping), CPU only. Exported to ONNX for the app."],
        ["3", "Landmark classifier", "Gradient boosting over 61 features (EAR, MAR, head pose and 52 blendshapes). Chosen on validation from logistic regression, random forest, gradient boosting and an MLP."],
        ["4", "Ensemble", "Weighted average of Run 2 and Run 3 probabilities. Weight and CNN input (whole image or face crop) chosen on validation."],
    ], [1.2, 4.4, 11.0])
    para(d, "Every model was selected using the validation split only. The test split was used once, for the "
            "final numbers below.")

    d.add_heading("5.3 Results on the test set", level=2)
    rows = []
    for r in summ:
        pc = lambda v: "–" if v is None else f"{100 * v:.1f}%"
        rows.append([r["run"].split(":")[0], r["run"].split(": ", 1)[1], pc(r["accuracy"]), pc(r["drowsy_recall"]),
                     pc(r["drowsy_precision"]), "–" if r.get("roc_auc") is None else f"{r['roc_auc']:.3f}"])
    table(d, ["Run", "Model", "Accuracy", "Drowsy recall", "Drowsy precision", "ROC-AUC"], rows,
          [1.4, 5.8, 2.3, 2.4, 2.6, 2.1])
    done = {r["run"].split(":")[0] for r in summ}
    pending = [r for r in ("Run 2", "Run 4") if r not in done]
    if pending:
        note(d, "Note:", f"{' and '.join(pending)} (CNN and ensemble) were still training when this document was "
                         "generated. Their results will be added to this table when training completes.")
    para(d, "Drowsy recall is the share of truly drowsy faces that were flagged, which is the most important "
            "number for safety. Drowsy precision is the share of alarms that were correct.")
    image(d, runs / "model_comparison.png", 15, "Figure: test-set comparison of the four runs")
    cnn_dir = runs / "classify" / "drowsiness_yolov8n_cls"
    image(d, cnn_dir / "results.png", 15, "Figure: CNN training curves (loss and validation accuracy per epoch)")
    best = max(summ, key=lambda r: r["accuracy"] or 0)
    para(d, f"The app uses the trained models' verdict ({'best so far' if pending else 'best'}: {best['run'].split(': ', 1)[1]}, "
            f"{100 * best['accuracy']:.1f}% test accuracy). The rule-based state is still shown as the explanation "
            "(eyes closed, yawning, nodding) and still drives the live monitor's duration timers.")


# =========================================================================== EXECUTION GUIDE
def build_execution():
    d = new_doc("Execution Guide",
                "Driver Drowsiness Detection: Installation, Running and Usage")

    d.add_heading("1. Purpose", level=1)
    para(d, "This document explains, step by step, how to install, run, use and test the Driver Drowsiness "
            "Detection web application on a new computer. Commands are given for Linux/macOS and Windows.")

    d.add_heading("2. System Requirements", level=1)
    table(d, ["Item", "Requirement"], [
        ["Operating system", "Windows 10/11, Ubuntu 20.04+ (or other Linux), macOS 11+"],
        ["Python", "3.10, 3.11 or 3.12 (64-bit)"],
        ["RAM / disk", "4 GB RAM minimum; ~1 GB free disk space for dependencies"],
        ["Camera", "Any built-in or USB webcam (only needed for Live Monitor)"],
        ["Browser", "Google Chrome, Microsoft Edge or Mozilla Firefox (latest version)"],
        ["Internet", "Needed once to install packages. The UI also loads Bootstrap/Chart.js from a CDN."],
        ["GPU", "Not required: everything runs on the CPU"],
    ], [4.2, 12.4])

    d.add_heading("3. Installation", level=1)
    d.add_heading("Step 1: Check Python", level=2)
    para(d, "Open a terminal (Command Prompt or PowerShell on Windows) and run:")
    code(d, "python --version          (Linux/macOS may need: python3 --version)")
    para(d, "The output must show Python 3.10 or newer. If Python is missing, install it from "
            "https://www.python.org/downloads/. On Windows, tick \"Add Python to PATH\" during installation.")

    d.add_heading("Step 2: Open the project folder", level=2)
    para(d, "Move into the project's root folder (the folder that contains README.md, ml_model and "
            "drowsiness_detection_webapp):")
    code(d, 'cd "path/to/DRIVER DROWSINESS DETECTION"')

    d.add_heading("Step 3: Create and activate a virtual environment", level=2)
    para(d, "A virtual environment keeps this project's packages separate from the rest of the system.")
    para(d, "", "Linux / macOS:")
    code(d, "python3 -m venv .venv\nsource .venv/bin/activate")
    para(d, "", "Windows (Command Prompt):")
    code(d, "python -m venv .venv\n.venv\\Scripts\\activate")
    para(d, "", "Windows (PowerShell):")
    code(d, ".venv\\Scripts\\Activate.ps1")
    note(d, "Tip:", "When the environment is active, the prompt starts with `(.venv)`. If PowerShell blocks the "
                    "script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.")

    d.add_heading("Step 4: Install dependencies", level=2)
    code(d, "pip install --upgrade pip\npip install -r drowsiness_detection_webapp/requirements.txt")
    para(d, "This installs Django, MediaPipe, OpenCV, NumPy, Pillow, scikit-learn and ReportLab. It may take a few "
            "minutes. The trained models are already included, so PyTorch is not needed to run the app.")

    d.add_heading("Step 5: Set up the database", level=2)
    code(d, "cd drowsiness_detection_webapp\npython manage.py migrate")
    para(d, "This creates the SQLite database file `db.sqlite3` with the required tables.")

    d.add_heading("Step 6 (optional): Create an admin user", level=2)
    code(d, "python manage.py createsuperuser")
    para(d, "Enter a username and password when prompted. You can then sign in at http://127.0.0.1:8000/admin/.")

    d.add_heading("4. Running the Application", level=1)
    para(d, "From inside the `drowsiness_detection_webapp` folder, with the virtual environment active:")
    code(d, "python manage.py runserver")
    para(d, "When you see \"Starting development server at http://127.0.0.1:8000/\", open that address in your browser.")
    note(d, "Important:", "use `http://127.0.0.1:8000` or `http://localhost:8000`. Browsers only allow webcam access "
                          "on localhost or HTTPS, so the Live Monitor will not work from another computer's IP address "
                          "over plain HTTP.")
    para(d, "To stop the server, press Ctrl + C in the terminal. To run on a different port, use "
            "`python manage.py runserver 8080`.")
    para(d, "The face landmark model `ml_model/models/face_landmarker.task` is already included. If it is "
            "deleted, the application downloads it automatically the first time it runs (internet required).")

    d.add_heading("5. Using the Application", level=1)
    d.add_heading("5.1 Live Monitor (webcam)", level=2)
    bullets(d, [
        "Click Live Monitor in the navigation bar.",
        "Click Start and allow camera access when the browser asks.",
        "Look at the camera with your eyes open for about 3 seconds while the status bar shows "
        "\"Calibrating\". The system learns your normal eye openness.",
        "The status bar then shows \"Driver alert\". Eye and mouth outlines are drawn over the video, and the "
        "panel shows live EAR, MAR, head pitch and session statistics.",
        "To test the alarm, close your eyes for 2 seconds. The border flashes red, a beep sounds and the status "
        "shows \"WAKE UP!\". A snapshot is saved to History automatically.",
        "Use the speaker button to mute or unmute the alarm, and Stop to end the session.",
    ])
    d.add_heading("5.2 Analyse an image", level=2)
    bullets(d, [
        "Click Upload in the navigation bar.",
        "Drag an image onto the upload area, or click it to choose a file (JPG, PNG or WEBP, maximum 10 MB). "
        "Use a clear photo of the driver's face.",
        "Click Detect Drowsiness. The result page shows the detected state, the annotated image (toggle "
        "Annotated/Original), an indicator chart and the measurements.",
        "Click Download Report to save a PDF of the result.",
    ])
    d.add_heading("5.3 History and reports", level=2)
    para(d, "The History page lists every uploaded image and every live-alarm snapshot with summary statistics. "
            "Click Details to reopen a result, or Report to download its PDF.")
    d.add_heading("5.4 How It Works page", level=2)
    para(d, "Shows whether the model loaded correctly, the detection thresholds and the live-monitor rules. "
            "Check this page first if detection does not seem to work.")

    d.add_heading("6. Using the JSON API", level=1)
    para(d, "Other programs can send an image and receive the analysis as JSON:")
    code(d, "curl -F image=@driver.jpg http://127.0.0.1:8000/api/predict/")
    para(d, "Example response (shortened):")
    code(d, """
{
  "success": true,
  "face_detected": true,
  "state_name": "Eyes closed",
  "is_drowsy": true,
  "confidence": 1.0,
  "drowsiness_score": 0.73,
  "metrics": {"ear": 0.037, "mar": 0.008, "pitch": 26.7, "blink": 0.69, "jaw_open": 0.0},
  "message": "DANGER: Driver's eyes are closed. Possible microsleep."
}
""")

    d.add_heading("7. Command-Line Usage (without the web server)", level=1)
    para(d, "From the project root folder:")
    code(d, "python ml_model/drowsiness_detector.py photo1.jpg photo2.jpg")
    para(d, "Each image's state and measurements are printed, and an annotated copy (`photo1_annotated.jpg`) "
            "is written next to the original.")

    d.add_heading("8. Running the Automated Tests", level=1)
    code(d, "cd drowsiness_detection_webapp\npython manage.py test detector\n\n(or from the project root: scripts/08_test.sh)")
    para(d, "Seven tests check that every page loads; that an upload produces a consistent result (state, "
            "verdict and model probability agree); that a closed-eye face is detected as drowsy; that the PDF "
            "report, the How It Works page and both APIs work. Expected output ends with:")
    code(d, "Ran 7 tests in 4.3s\n\nOK")

    d.add_heading("9. Reproducing the Training Runs (optional)", level=1)
    para(d, "The trained models are already included in `ml_model/models/`, so this step is only needed to "
            "retrain or verify them. It downloads about 840 MB and needs about 2 GB of free disk space. On a "
            "20-core CPU the whole pipeline takes about 45 minutes, most of it CNN training. No GPU is needed.")
    para(d, "", "Quickest way: bash scripts (Linux/macOS, from the project root):")
    code(d, "scripts/00_setup.sh --train --docs    (use --gpu instead of --train on an NVIDIA GPU)\n"
            "scripts/run_all.sh                    (steps 1-8; --from N to resume, --skip-cnn to skip the CNN)\n"
            "scripts/status.sh                     (shows which steps are done / running / to do)")
    para(d, "To use a GPU for the CNN step: `DEVICE=0 scripts/03_run2_cnn.sh` (about 3–6 minutes instead of "
            "about 37). RUNS_LINEUP.md lists every step with its inputs, outputs and timings.")
    para(d, "", "Manual way (any OS). Install the training packages (from the project root, virtual environment active):")
    code(d, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "pip install -r training/requirements.txt")
    para(d, "", "Then run the steps in order:")
    table(d, ["#", "Command", "What it does", "Time"], [
        ["1", "`python training/prepare_dataset.py`", "Downloads the dataset, removes leakage and eye-only crops, writes datasets/driver_drowsiness/", "~3 min"],
        ["2", "`python training/extract_landmark_features.py`", "Landmark features for every image + Run 1 (rule-based) evaluation", "~1 min"],
        ["3", "`python training/train_cnn.py`", "Run 2: trains YOLOv8n-cls, evaluates on test, exports ONNX to ml_model/models/", "~35 min"],
        ["4", "`python training/train_landmark_classifier.py`", "Run 3: compares 4 classifiers, saves the best", "~20 s"],
        ["5", "`python training/evaluate_ensemble.py`", "Run 4: chooses the ensemble on validation, evaluates on test", "~1 min"],
        ["6", "`python training/summarize_runs.py`", "Writes runs/summary.md, runs/summary.json and runs/model_comparison.png", "seconds"],
    ], [0.8, 7.8, 6.2, 1.8])
    para(d, "Results are written to the `runs/` folder: per-run `test_metrics.json` files, CNN training curves "
            "and confusion matrices in `runs/classify/drowsiness_yolov8n_cls/`, and logs in `runs/logs/`. Restart "
            "the web server afterwards so it loads the new models.")

    d.add_heading("10. Configuration", level=1)
    table(d, ["Setting", "Location", "Default"], [
        ["EAR closed threshold", "`ml_model/drowsiness_detector.py`: `EAR_CLOSED`", "0.21"],
        ["Yawn threshold (MAR)", "`ml_model/drowsiness_detector.py`: `MAR_YAWN`", "0.60"],
        ["Head-nod angle", "`ml_model/drowsiness_detector.py`: `PITCH_NOD_DEG`", "20°"],
        ["Eyes-closed alarm time", "`templates/detector/live.html`: `CFG.closedAlarmMs`", "1500 ms"],
        ["Head-nod alarm time", "`templates/detector/live.html`: `CFG.nodAlarmMs`", "2000 ms"],
        ["PERCLOS warning level", "`templates/detector/live.html`: `CFG.perclosWarn`", "0.15 (15%)"],
        ["Analysis rate", "`templates/detector/live.html`: `CFG.intervalMs`", "140 ms (~7 fps)"],
        ["Ensemble weight / threshold", "`ml_model/models/ensemble.json`", "chosen on validation; 0.5"],
        ["Model-fatigue warning", "`templates/detector/live.html`: `CFG.modelWarn`", "0.7 for 5 s"],
        ["Maximum upload size", "`drowsiness_detection_web/settings.py`", "10 MB"],
    ], [4.4, 8.6, 3.6])
    para(d, "Restart the server after changing Python settings. Refresh the browser after changing live.html.")

    d.add_heading("11. Troubleshooting", level=1)
    table(d, ["Problem", "Cause / Solution"], [
        ["`python` not found", "Install Python 3.10+ and make sure it is on PATH. On Linux/macOS, try `python3`."],
        ["`No module named django` (or mediapipe / cv2)", "The virtual environment is not active. Activate it (Step 3) and reinstall requirements (Step 4)."],
        ["MediaPipe fails to install", "Use 64-bit Python 3.10–3.12. Very new Python versions may not have MediaPipe wheels yet."],
        ["Camera does not start / \"Camera unavailable\"", "Open the site via `127.0.0.1` or `localhost`, allow camera permission in the browser, and close other apps (Zoom, Teams) that are using the camera."],
        ["Stuck on \"Looking for face...\"", "Face the camera directly, improve lighting, remove dark sunglasses, move closer."],
        ["Too many false \"eyes closed\" alarms", "Restart the monitor and keep eyes fully open during calibration, or lower `EAR_CLOSED` slightly."],
        ["\"No face detected\" on an uploaded photo", "Use a clearer, more frontal photo where the face is reasonably large."],
        ["Port 8000 already in use", "Stop the other server or run `python manage.py runserver 8080`."],
        ["\"Model unavailable\" on How It Works page", "`face_landmarker.task` is missing and could not be downloaded. Check the internet connection, or download it manually into `ml_model/models/`."],
        ["\"Trained models: not found\" on How It Works page", "`drowsiness_cls.onnx` or `landmark_classifier.joblib` is missing from `ml_model/models/`. Restore them or re-run the training (section 9). The app still works with rule-based detection only."],
        ["`No module named sklearn`", "Install the requirements again: `pip install -r drowsiness_detection_webapp/requirements.txt`."],
        ["Page looks unstyled", "Bootstrap and icons load from a CDN. Check the internet connection."],
    ], [5.6, 11.0])

    d.add_heading("12. Quick Command Reference", level=1)
    table(d, ["Task", "Command"], [
        ["Activate environment (Linux/macOS)", "`source .venv/bin/activate`"],
        ["Activate environment (Windows)", "`.venv\\Scripts\\activate`"],
        ["Install packages", "`pip install -r drowsiness_detection_webapp/requirements.txt`"],
        ["Create database", "`python manage.py migrate`"],
        ["Create admin user", "`python manage.py createsuperuser`"],
        ["Start server", "`python manage.py runserver` (or `scripts/run_server.sh`)"],
        ["Pipeline status", "`scripts/status.sh`"],
        ["Run tests", "`python manage.py test detector`"],
        ["Analyse images from CLI", "`python ml_model/drowsiness_detector.py image.jpg`"],
        ["Re-run all training (from project root)", "`scripts/run_all.sh` (or the Python commands in section 9)"],
        ["Stop server", "Ctrl + C"],
    ], [6.0, 10.6])

    d.save(OUT / "EXECUTION_GUIDE.docx")


if __name__ == "__main__":
    build_readme()
    build_execution()
    print(f"Wrote {OUT / 'README.docx'} and {OUT / 'EXECUTION_GUIDE.docx'}")
