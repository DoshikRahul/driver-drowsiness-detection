#!/usr/bin/env bash
# Show the pipeline lineup and the status of every step, based on the files each step produces.
source "$(dirname "$0")/common.sh"

done_if() { local f; for f in "$@"; do [[ -e "$ROOT/$f" ]] || return 1; done; }
running() { pgrep -f "$1" >/dev/null; }

row() {  # row <num> <name> <script> <status>
    local col="$c_red"
    case "$4" in DONE*) col="$c_green" ;; RUNNING*) col="$c_yellow" ;; WAITING*|TODO*) col="$c_off" ;; esac
    printf '  %-3s %-34s %-36s %s%s%s\n' "$1" "$2" "$3" "$col" "$4" "$c_off"
}

status_of() {  # status_of <running-pattern|-> <outputs...>
    local pat="$1"; shift
    if [[ "$pat" != "-" ]] && running "$pat"; then echo "RUNNING"; return; fi
    if done_if "$@"; then echo "DONE"; else echo "TODO"; fi
}

cnn_status="$(status_of training/train_cnn.py runs/classify/drowsiness_yolov8n_cls/test_metrics.json ml_model/models/drowsiness_cls.onnx)"
if [[ "$cnn_status" == RUNNING && -f runs/classify/drowsiness_yolov8n_cls/results.csv ]]; then
    last="$(tail -1 runs/classify/drowsiness_yolov8n_cls/results.csv)"
    [[ "$last" == epoch* ]] || cnn_status="RUNNING (epoch ${last%%,*} done, val acc $(echo "$last" | cut -d, -f4))"
fi
ens_status="$(status_of training/evaluate_ensemble.py runs/ensemble/test_metrics.json ml_model/models/ensemble.json)"
[[ "$ens_status" == TODO && ! -e ml_model/models/drowsiness_cls.onnx ]] && ens_status="WAITING for step 3"

echo
echo "  #   Step                               Script                               Status"
echo "  --- ---------------------------------- ------------------------------------ ------"
row 0 "Setup environment"            "scripts/00_setup.sh"                "$( [[ -x "$PY" ]] && echo DONE || echo TODO)"
row 1 "Prepare dataset"              "scripts/01_prepare_dataset.sh"      "$(status_of training/prepare_dataset.py runs/dataset_report.json datasets/driver_drowsiness)"
row 2 "Run 1: rule-based baseline"   "scripts/02_run1_rule_based.sh"      "$(status_of training/extract_landmark_features.py runs/landmark/features_test.npz runs/rule_based/test_metrics.json)"
row 3 "Run 2: CNN (YOLOv8n-cls)"     "scripts/03_run2_cnn.sh"             "$cnn_status"
row 4 "Run 3: landmark classifier"   "scripts/04_run3_landmark_classifier.sh" "$(status_of training/train_landmark_classifier.py runs/landmark/test_metrics.json ml_model/models/landmark_classifier.joblib)"
row 5 "Run 4: ensemble"              "scripts/05_run4_ensemble.sh"        "$ens_status"
row 6 "Summarize runs"               "scripts/06_summarize.sh"            "$(status_of - runs/summary.json runs/model_comparison.png)"
row 7 "Build documents"              "scripts/07_build_docs.sh"           "$(status_of - README.docx EXECUTION_GUIDE.docx)"
row 8 "Automated tests"              "scripts/08_test.sh"                 "$( [[ -f runs/logs/tests.log ]] && (grep -q '^OK' runs/logs/tests.log && echo "DONE (last run passed)" || echo "FAILED last run") || echo TODO)"
echo

if [[ -f runs/summary.md ]]; then
    echo "  Latest results (runs/summary.md):"
    grep '^| Run' runs/summary.md | sed 's/^/    /'
    if [[ runs/summary.json -ot runs/classify/drowsiness_yolov8n_cls/test_metrics.json ]] 2>/dev/null \
       || [[ -e runs/ensemble/test_metrics.json && runs/summary.json -ot runs/ensemble/test_metrics.json ]]; then
        warn "Summary is older than the latest run results: run scripts/06_summarize.sh and scripts/07_build_docs.sh"
    fi
    echo
fi
