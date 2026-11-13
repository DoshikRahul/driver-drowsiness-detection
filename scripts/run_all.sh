#!/usr/bin/env bash
# Run the whole pipeline in order (steps 1-8), stopping at the first failure.
#
#   scripts/run_all.sh                     everything (CPU: ~45 min)
#   scripts/run_all.sh --from 5            resume from step 5
#   scripts/run_all.sh --from 3 --to 5     only steps 3..5
#   scripts/run_all.sh --skip-cnn          skip steps 3 (CNN) and 5 (ensemble)
#   DEVICE=0 scripts/run_all.sh            train the CNN on an NVIDIA GPU
#
# Steps:
#   1 prepare dataset          5 run 4: ensemble
#   2 run 1: rule-based        6 summarize runs
#   3 run 2: CNN               7 build documents
#   4 run 3: landmark clf      8 tests
source "$(dirname "$0")/common.sh"

FROM=1; TO=8; SKIP_CNN=0
while (( $# )); do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --skip-cnn) SKIP_CNN=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) die "Unknown option $1" ;;
    esac
done

STEPS=(
    ""                                   # (1-based)
    01_prepare_dataset.sh
    02_run1_rule_based.sh
    03_run2_cnn.sh
    04_run3_landmark_classifier.sh
    05_run4_ensemble.sh
    06_summarize.sh
    07_build_docs.sh
    08_test.sh
)

TOTAL_START=$SECONDS
for (( i = FROM; i <= TO; i++ )); do
    if (( SKIP_CNN )) && (( i == 3 || i == 5 )); then
        warn "Skipping step $i (${STEPS[$i]})"
        continue
    fi
    echo
    bash "$ROOT/scripts/${STEPS[$i]}"
done
echo
ok "Pipeline steps $FROM-$TO finished in $(( (SECONDS - TOTAL_START) / 60 ))m $(( (SECONDS - TOTAL_START) % 60 ))s"
bash "$ROOT/scripts/status.sh"
