#!/usr/bin/env bash
# Step 6: collect all test metrics into runs/summary.{json,md} and runs/model_comparison.png
source "$(dirname "$0")/common.sh"
require_venv
require_python_module matplotlib "Run scripts/00_setup.sh --train"
require_file runs/dataset_report.json "Run scripts/01_prepare_dataset.sh first."
run_step "Step 6: summarize runs" summarize_runs.log "$PY" training/summarize_runs.py
