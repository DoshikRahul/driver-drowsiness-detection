#!/usr/bin/env bash
# Step 8: run the Django test suite (pages, upload, models, APIs, PDF report).
source "$(dirname "$0")/common.sh"
require_venv
run_step "Step 8: automated tests" tests.log bash -c "cd drowsiness_detection_webapp && '$PY' manage.py test detector -v 1"
