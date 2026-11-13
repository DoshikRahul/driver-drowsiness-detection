#!/usr/bin/env bash
# Step 1: download the dataset (~840 MB), remove split leakage and eye-only crops,
# write datasets/driver_drowsiness/{train,valid,test}/{drowsy,not_drowsy}.
#   scripts/01_prepare_dataset.sh [--force]     (--force rebuilds an existing dataset)
source "$(dirname "$0")/common.sh"
require_venv
require_python_module pyarrow "Run scripts/00_setup.sh --train"
run_step "Step 1: prepare dataset" prepare_dataset.log "$PY" training/prepare_dataset.py "$@"
