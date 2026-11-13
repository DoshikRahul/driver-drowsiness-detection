# Driver Drowsiness Detection Using Computer Vision and Deep Learning

> Full documentation: **README.docx** (project description) and **EXECUTION_GUIDE.docx** (installation, running, usage and troubleshooting).

A Django web app that detects driver drowsiness from a webcam stream or an uploaded photo.
It finds the driver's face with **MediaPipe Face Landmarker** (478 landmarks + 52 blendshapes) and measures three fatigue signals:

| Signal | How | Drowsy when |
|---|---|---|
| **Eye Aspect Ratio (EAR)** | `(\|p2−p6\| + \|p3−p5\|) / (2·\|p1−p4\|)` over 6 eye landmarks, fused with the `eyeBlink` blendshape | EAR < 0.21 (or calibrated) / blink > 0.5 |
| **Mouth Aspect Ratio (MAR)** | inner-lip opening ÷ mouth width, fused with the `jawOpen` blendshape | MAR > 0.60 / jawOpen > 0.5 → yawning |
| **Head pitch** | from the 3-D facial transformation matrix | pitched down > 20° → nodding |

Each frame is classified as **Alert**, **Eyes closed**, **Yawning**, **Head nodding** or **No face detected**.

On top of these rules, two models trained on a public drowsiness dataset give the final drowsy / not-drowsy verdict:
- a **CNN** (YOLOv8n-cls, fine-tuned, run with OpenCV DNN from ONNX), and
- a **gradient-boosting classifier** over the landmark features.

They are combined as a weighted average chosen on validation data. If the models flag drowsiness but no single rule fired, the state is **Fatigue signs**.

## Dataset and training runs
Dataset: [n7i5x9/driver-drowsiness-dataset](https://huggingface.co/datasets/n7i5x9/driver-drowsiness-dataset), with 23,116 images labelled drowsy / not_drowsy. Two problems were cleaned up before training:
- **Split leakage:** the published splits share augmented copies of the same photos (≈45% of test images also appear in train). The data were re-split by original photo (80/10/10), and remaining duplicates were removed by perceptual hash.
- **Eye-only crops:** 9,302 images are close-ups of a single eye, not faces, and were excluded.

That leaves 10,973 train / 739 validation / 745 test face images. Results on the untouched test split:

| Run | Accuracy | Drowsy recall | Drowsy precision | ROC-AUC |
|---|---|---|---|---|
| 1. Rule-based landmarks | 65.5% | 77.5% | 63.5% | – |
| 2. CNN (YOLOv8n-cls) | 88.5% | 94.8% | 84.6% | 0.940 |
| 3. Landmark features + gradient boosting | 87.0% | 91.6% | 84.4% | 0.916 |
| 4. Ensemble (0.8 × CNN + 0.2 × landmark), **used by the app** | **88.6%** | **95.3%** | 84.5% | **0.945** |

| Run | Method | Script |
|---|---|---|
| 1 | Rule-based landmarks (baseline) | `training/extract_landmark_features.py` |
| 2 | CNN: YOLOv8n-cls fine-tuned | `training/train_cnn.py` |
| 3 | Landmark features + gradient boosting | `training/train_landmark_classifier.py` |
| 4 | Ensemble of runs 2 + 3 | `training/evaluate_ensemble.py` |

Reproduce everything with the bash scripts in `scripts/` (CPU ~45 min; see **RUNS_LINEUP.md** for every step):
```bash
scripts/00_setup.sh --train --docs   # venv + all packages (use --gpu for CUDA)
scripts/run_all.sh                   # steps 1-8: dataset, 4 runs, summary, docs, tests
scripts/status.sh                    # live status of each step + latest results
scripts/finish_pending.sh            # wait for a running CNN, then ensemble/summary/docs/tests
DEVICE=0 scripts/03_run2_cnn.sh      # train the CNN on an NVIDIA GPU
scripts/run_server.sh                # start the web app
```

### Live monitor (webcam)
The browser sends about 7 frames/s to the server. The page then applies time-based rules:
- **Calibration:** for the first ~3 s it learns the driver's normal open-eye EAR and sets the threshold to 75% of it.
- **Alarm:** sounds when the eyes are closed for ≥ 1.5 s or the head nods for ≥ 2 s. A snapshot is saved to History.
- **PERCLOS:** the percentage of time the eyes were closed over the last 60 s. Above 15% raises a fatigue warning.
- **Yawn counter:** 3 or more yawns in 5 minutes raises a fatigue warning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r drowsiness_detection_webapp/requirements.txt

cd drowsiness_detection_webapp
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/. The webcam only works on `localhost` or over HTTPS.

The face landmark model (`ml_model/models/face_landmarker.task`, ~3.7 MB) is included. If it is missing, it is downloaded automatically on first use.

Or use the scripts: `scripts/00_setup.sh` then `scripts/run_server.sh`.

The trained models the app needs are in the repository. The dataset (~1.3 GB) is not; `scripts/01_prepare_dataset.sh` downloads and rebuilds it if you want to re-train (see [RUNS_LINEUP.md](RUNS_LINEUP.md)).

For anything other than local use, set these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | a development key | Must be set to a random secret when deployed |
| `DJANGO_DEBUG` | `1` | Set to `0` in production |
| `DJANGO_ALLOWED_HOSTS` | empty (localhost only) | Comma-separated host names |

## Usage
- **Live Monitor:** click *Start*, allow camera access and keep your eyes open while it calibrates.
- **Upload:** analyse a single photo. You get an annotated image, the measurements and a PDF report.
- **History:** past uploads and live-alarm snapshots.
- **API:**
  ```bash
  curl -F image=@face.jpg http://127.0.0.1:8000/api/predict/
  ```

## Command-line
```bash
python ml_model/drowsiness_detector.py photo1.jpg photo2.jpg   # writes *_annotated.jpg
```

## Tests
```bash
cd drowsiness_detection_webapp && python manage.py test detector
```

## Project layout
```
ml_model/drowsiness_detector.py       core detector (EAR / MAR / head pose)
ml_model/drowsiness_models.py         trained CNN + landmark classifier (inference)
ml_model/models/                      face_landmarker.task, drowsiness_cls.onnx, landmark_classifier.joblib, ensemble.json
training/                             dataset preparation + training / evaluation scripts
scripts/                              bash scripts: setup, each pipeline step, run_all, status, server
docs/                                 make_docs.py (builds the .docx files) + screenshots
datasets/driver_drowsiness/           prepared train / valid / test images (not in git; built by scripts/01_prepare_dataset.sh)
runs/                                 metrics, curves, confusion matrices, summary
drowsiness_detection_webapp/          Django project
  detector/ml_service.py              thread-safe wrapper used by the views
  detector/templates/detector/live.html   webcam monitor + temporal logic
legacy_distracted_driver/             scripts from the previous YOLOv8 distracted-driver version (no longer used)
```

## Limitations
- Dark sunglasses hide the eyes, so eye closure can't be measured through them.
- Strong profile views (> ~45° yaw) reduce accuracy.
- Looking down (e.g. at a phone) lowers the eyelids and can read as "eyes closed" in a single photo. The live monitor's duration rules reduce such false alarms.
- The training images are mostly still photos from the web, not in-car video, and the same person may appear in more than one split. Expect real-world accuracy to be lower than the test-set figures.
- **Dataset bias:** every in-car photo in the dataset is labelled *drowsy*; *not-drowsy* images are almost all close-up faces, and the test set shares this bias. The trained models can therefore over-report drowsiness for alert drivers photographed in a car, especially when talking (an open mouth looks like a yawn). The live-monitor alarm stays duration-based for this reason.
