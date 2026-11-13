#!/usr/bin/env bash
# Step 3 / Run 2: fine-tune YOLOv8n-cls, evaluate on test, export ONNX to ml_model/models/.
# Output: runs/classify/drowsiness_yolov8n_cls/, ml_model/models/drowsiness_cls.onnx (+ .json)
#
#   scripts/03_run2_cnn.sh                    CPU (~35 min on 20 cores)
#   DEVICE=0 scripts/03_run2_cnn.sh           NVIDIA GPU (~3-6 min); uses batch 256 + RAM cache
#   EPOCHS=30 IMGSZ=224 scripts/03_run2_cnn.sh
source "$(dirname "$0")/common.sh"
require_venv
require_python_module ultralytics "Run scripts/00_setup.sh --train (or --gpu)"
require_file datasets/driver_drowsiness "Run scripts/01_prepare_dataset.sh first."

if pgrep -f "training/train_cnn.py" >/dev/null; then
    die "A CNN training run is already in progress (pid $(pgrep -f training/train_cnn.py | head -1))."
fi

DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-20}"
IMGSZ="${IMGSZ:-224}"
if [[ "$DEVICE" == "cpu" ]]; then
    BATCH="${BATCH:-64}"; EXTRA=(--threads "${THREADS:-12}" --workers "${WORKERS:-6}")
else
    "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || die "DEVICE=$DEVICE requested but PyTorch sees no CUDA GPU. Run scripts/00_setup.sh --gpu"
    BATCH="${BATCH:-256}"; EXTRA=(--workers "${WORKERS:-8}" --cache)
fi

run_step "Run 2: CNN training on $DEVICE ($EPOCHS epochs, batch $BATCH)" train_cnn.log \
    "$PY" training/train_cnn.py --device "$DEVICE" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" "${EXTRA[@]}"
