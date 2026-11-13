#!/usr/bin/env bash
# Step 4 / Run 3: train and compare classifiers on the landmark features; save the best.
# Output: runs/landmark/test_metrics.json, runs/landmark/model_comparison.json,
#         ml_model/models/landmark_classifier.joblib
source "$(dirname "$0")/common.sh"
require_venv
require_file runs/landmark/features_test.npz "Run scripts/02_run1_rule_based.sh first."
run_step "Run 3: landmark classifier" train_landmark_classifier.log "$PY" training/train_landmark_classifier.py
