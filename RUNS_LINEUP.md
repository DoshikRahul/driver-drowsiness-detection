# Pipeline Lineup: Driver Drowsiness Detection

Every step, in order: what it needs, what it produces and how to run it.
Run `scripts/status.sh` at any time to see live status (it checks the files each step produces).

```
00 setup ─► 01 dataset ─┬─► 02 Run 1 rule-based ─► 04 Run 3 landmark clf ─┐
                        │      (landmark features)                         ├─► 05 Run 4 ensemble ─► 06 summary ─► 07 docs ─► 08 tests
                        └─► 03 Run 2 CNN (YOLOv8n-cls) ─────────────────────┘
```

## Steps

| # | Step | Script | Needs | Produces | CPU time* | GPU time** |
|---|---|---|---|---|---|---|
| 0 | Environment | `scripts/00_setup.sh [--train] [--gpu] [--docs]` | Python 3.10–3.12 | `.venv/`, database | 3–10 min | same |
| 1 | Prepare dataset | `scripts/01_prepare_dataset.sh [--force]` | internet (~840 MB) | `datasets/driver_drowsiness/`, `runs/dataset_report.json` | ~3 min | same |
| 2 | Run 1: rule-based baseline | `scripts/02_run1_rule_based.sh` | step 1 | `runs/landmark/features_*.npz`, `runs/rule_based/test_metrics.json` | ~1 min | same (CPU) |
| 3 | Run 2: CNN (YOLOv8n-cls) | `scripts/03_run2_cnn.sh` (`DEVICE=0` for GPU) | step 1, `--train` packages | `runs/classify/drowsiness_yolov8n_cls/`, `ml_model/models/drowsiness_cls.onnx` + `.json` | ~37 min | ~3–6 min |
| 4 | Run 3: landmark classifier | `scripts/04_run3_landmark_classifier.sh` | step 2 | `runs/landmark/test_metrics.json`, `ml_model/models/landmark_classifier.joblib` | ~20 s | same |
| 5 | Run 4: ensemble | `scripts/05_run4_ensemble.sh` | steps 3 + 4 | `runs/ensemble/test_metrics.json`, `ml_model/models/ensemble.json` | ~1 min | same |
| 6 | Summary | `scripts/06_summarize.sh` | any finished runs | `runs/summary.{json,md}`, `runs/model_comparison.png` | seconds | same |
| 7 | Documents | `scripts/07_build_docs.sh` | step 6, `--docs` packages | `README.docx`, `EXECUTION_GUIDE.docx` (+ PDFs in `docs/build/` if LibreOffice is installed) | seconds | same |
| 8 | Tests | `scripts/08_test.sh` | step 0 | `runs/logs/tests.log` | ~5 s | same |

\* Measured on this machine (20-core CPU, no GPU). \*\* Estimated for an NVIDIA RTX PRO 4500 Blackwell-class GPU. Only step 3 uses the GPU.

All logs go to `runs/logs/<step>.log`. Every script checks its prerequisites and stops with a clear message if one is missing.

## Orchestration scripts

| Script | Purpose |
|---|---|
| `scripts/run_all.sh` | Run steps 1–8 in order. Options: `--from N`, `--to N`, `--skip-cnn`. `DEVICE=0` trains the CNN on a GPU. |
| `scripts/finish_pending.sh` | Waits for a running CNN training (prints each epoch), checks its outputs, then runs steps 5–8. |
| `scripts/status.sh` | Shows this lineup with live DONE / RUNNING / TODO status and the latest results table. |
| `scripts/run_server.sh [port]` | Starts the web app on http://127.0.0.1:8000/. |

## Current status

All steps finished on 2026-11-13 (CPU, 20 cores). Total CNN training time: 35 min, 20 epochs.

| Run | Accuracy | Drowsy recall | Drowsy precision | ROC-AUC |
|---|---|---|---|---|
| 1. Rule-based landmarks | 65.5% | 77.5% | 63.5% | – |
| 2. CNN, YOLOv8n-cls | 88.5% | 94.8% | 84.6% | 0.940 |
| 3. Landmark features + gradient boosting | 87.0% | 91.6% | 84.4% | 0.916 |
| 4. Ensemble 0.8 × CNN + 0.2 × landmark (**used by the app**) | **88.6%** | **95.3%** | 84.5% | **0.945** |

Checks done: the ONNX/OpenCV model used by the app reproduces the PyTorch test accuracy exactly (88.46%), and all 7 tests pass.

### Known issue found after training

Every in-car photo in the dataset is labelled *drowsy* (249 in train, 0 in not_drowsy), and *not_drowsy* is almost all close-up faces. The test set has the same bias, so the scores above cannot reveal it. Example: an alert woman talking on the phone in a car (`detector/test_data/driver_alert.webp`) gets P(drowsy) = 0.99 from the CNN. A face-only crop doesn't help (0.96), because an open mouth while talking resembles a yawn. Mitigations already in place: the live-monitor alarm is duration-based (rules), and the model only adds a sustained-fatigue warning.

## Recommended next runs

1. **Collect alert in-car images** (your own webcam/dashcam, or a dataset such as NTHU-DDD / UTA-RLDD with subject-wise splits). Add them to `datasets/driver_drowsiness/*/not_drowsy`, then run `scripts/run_all.sh --from 2`. This is the most important fix.
2. **Evaluate on real webcam footage** of yourself (alert, talking, yawning, eyes closed) and record per-frame results before trusting the model's verdict.
3. If false alarms remain, raise the ensemble `threshold` in `ml_model/models/ensemble.json` (e.g. 0.7) using that footage. Don't tune it on the biased test set.

## Optional follow-up runs

- **GPU re-train:** `scripts/00_setup.sh --gpu`, then `DEVICE=0 EPOCHS=30 scripts/03_run2_cnn.sh` and `scripts/run_all.sh --from 5`.
- **Larger CNN** (e.g. YOLOv8s-cls): change `BASE_WEIGHTS` in `training/train_cnn.py`. This is practical only on a GPU.
- **Tune thresholds** (`EAR_CLOSED`, `MAR_YAWN`, `PITCH_NOD_DEG` in `ml_model/drowsiness_detector.py`), then re-run steps 2 and 6–8.
- **Fresh dataset rebuild:** `scripts/01_prepare_dataset.sh --force`, then `scripts/run_all.sh --from 2`. Every model must be retrained, because the splits are regenerated.
