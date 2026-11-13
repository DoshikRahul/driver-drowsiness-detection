#!/usr/bin/env bash
# Step 2 / Run 1: extract MediaPipe landmark features for every image and evaluate
# the rule-based detector on the test split.
# Output: runs/landmark/features_*.npz, runs/rule_based/test_metrics.json
#   WORKERS=6 scripts/02_run1_rule_based.sh
source "$(dirname "$0")/common.sh"
require_venv
require_python_module sklearn "Run scripts/00_setup.sh"
require_file datasets/driver_drowsiness "Run scripts/01_prepare_dataset.sh first."
run_step "Run 1: landmark features + rule-based evaluation" extract_landmark_features.log \
    "$PY" training/extract_landmark_features.py --workers "${WORKERS:-6}"
