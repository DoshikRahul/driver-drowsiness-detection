#!/usr/bin/env bash
# Step 5 / Run 4: choose CNN input + ensemble weight on validation, evaluate on test.
# Output: runs/ensemble/test_metrics.json, ml_model/models/ensemble.json
source "$(dirname "$0")/common.sh"
require_venv
require_file ml_model/models/drowsiness_cls.onnx "Run scripts/03_run2_cnn.sh first (or wait for it to finish)."
require_file ml_model/models/landmark_classifier.joblib "Run scripts/04_run3_landmark_classifier.sh first."
run_step "Run 4: ensemble" evaluate_ensemble.log "$PY" training/evaluate_ensemble.py
