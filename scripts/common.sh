#!/usr/bin/env bash
# Shared helpers for all pipeline scripts. Source it, don't run it.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"
LOGS="$ROOT/runs/logs"
mkdir -p "$LOGS"

# Quieten MediaPipe / TensorFlow-Lite C++ logging
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1

c_blue=$'\e[1;34m'; c_green=$'\e[1;32m'; c_red=$'\e[1;31m'; c_yellow=$'\e[1;33m'; c_off=$'\e[0m'
info()  { echo "${c_blue}==>${c_off} $*"; }
ok()    { echo "${c_green}✔${c_off}  $*"; }
warn()  { echo "${c_yellow}!${c_off}  $*"; }
die()   { echo "${c_red}✘  $*${c_off}" >&2; exit 1; }

require_venv() {
    [[ -x "$PY" ]] || die "Virtual environment not found at $VENV. Run scripts/00_setup.sh first."
}

require_python_module() {  # require_python_module <module> <hint>
    "$PY" -c "import $1" 2>/dev/null || die "Python package '$1' missing. $2"
}

require_file() {  # require_file <path> <hint>
    [[ -e "$ROOT/$1" ]] || die "Missing $1. $2"
}

# run_step <name> <log-file> <command...>: runs a command, tees output to a log, prints duration
run_step() {
    local name="$1" log="$LOGS/$2"; shift 2
    info "$name"
    local start=$SECONDS
    # Filter MediaPipe/TFLite log noise and its harmless shutdown traceback (sed always exits 0,
    # so with pipefail the step's status is the command's own exit code).
    if "$@" 2>&1 | sed -u -E \
        -e '/^(W0000|I0000|INFO: Created|WARNING: All log|WARNING: Logging before)/d' \
        -e '/^Exception ignored in: <function FaceLandmarker/,/^TypeError: .NoneType. object is not callable/d' \
        | tee "$log"; then
        ok "$name finished in $(( (SECONDS - start) / 60 ))m $(( (SECONDS - start) % 60 ))s  (log: ${log#$ROOT/})"
    else
        die "$name failed. See ${log#$ROOT/}"
    fi
}

cd "$ROOT"
