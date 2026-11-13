#!/usr/bin/env bash
# Create the virtual environment and install dependencies.
#
#   scripts/00_setup.sh            web app only (enough to run the server and tests)
#   scripts/00_setup.sh --train    + PyTorch (CPU), Ultralytics etc. to re-run training
#   scripts/00_setup.sh --gpu      + PyTorch with CUDA instead of CPU (for an NVIDIA GPU)
#   scripts/00_setup.sh --docs     + python-docx to regenerate README.docx / EXECUTION_GUIDE.docx
#   (flags can be combined)
source "$(dirname "$0")/common.sh"

TRAIN=0; GPU=0; DOCS=0
for a in "$@"; do
    case "$a" in
        --train) TRAIN=1 ;;
        --gpu)   TRAIN=1; GPU=1 ;;
        --docs)  DOCS=1 ;;
        *) die "Unknown option $a" ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x "$PY" ]]; then
    info "Creating virtual environment in $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
fi
"$PY" -m pip install -q --upgrade pip

info "Installing web-app requirements"
"$PY" -m pip install -q -r drowsiness_detection_webapp/requirements.txt

if (( TRAIN )); then
    if (( GPU )); then
        info "Installing PyTorch with CUDA (cu128)"
        "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
    else
        info "Installing PyTorch (CPU build)"
        "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    info "Installing training requirements"
    "$PY" -m pip install -q -r training/requirements.txt
fi

if (( DOCS )); then
    info "Installing documentation requirements"
    "$PY" -m pip install -q -r docs/requirements.txt
fi

info "Applying database migrations"
(cd drowsiness_detection_webapp && "$PY" manage.py migrate --noinput >/dev/null)

"$PY" - <<'EOF'
import importlib
for m in ["django", "mediapipe", "cv2", "sklearn", "reportlab", "torch", "ultralytics", "docx"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  {m:12s} (not installed)")
try:
    import torch
    print("  CUDA available:", torch.cuda.is_available())
except ImportError:
    pass
EOF
ok "Setup complete"
