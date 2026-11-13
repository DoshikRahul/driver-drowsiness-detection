#!/usr/bin/env bash
# Finish everything that is still pending after a CNN training run:
#   wait for train_cnn.py (if running) -> check its outputs -> run 4 (ensemble)
#   -> summary -> documents -> tests
#
#   scripts/finish_pending.sh              (safe to start while the CNN is still training)
source "$(dirname "$0")/common.sh"
require_venv

RESULTS="runs/classify/drowsiness_yolov8n_cls/results.csv"
if pgrep -f "training/train_cnn.py" >/dev/null; then
    info "CNN training is running, waiting for it to finish (checking every 30 s)..."
    last=""
    while pgrep -f "training/train_cnn.py" >/dev/null; do
        if [[ -f "$RESULTS" ]]; then
            line="$(tail -1 "$RESULTS")"
            if [[ "$line" != "$last" && "$line" != epoch* ]]; then
                IFS=, read -r ep secs _ acc _ <<<"$line"
                awk -v e="$ep" -v a="$acc" -v s="$secs" 'BEGIN { printf "   epoch %s done, val accuracy %.1f%%, elapsed %dm\n", e, a * 100, s / 60 }'
                last="$line"
            fi
        fi
        sleep 30
    done
    ok "CNN training process has exited"
fi

# The training process must have produced its evaluation and the exported model.
for f in runs/classify/drowsiness_yolov8n_cls/test_metrics.json ml_model/models/drowsiness_cls.onnx; do
    [[ -e "$f" ]] || die "$f is missing: CNN training did not complete. Check runs/logs/train_cnn.log, then re-run scripts/03_run2_cnn.sh"
done
ok "CNN outputs present"

bash "$ROOT/scripts/run_all.sh" --from 5 --to 8
