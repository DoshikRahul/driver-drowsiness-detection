#!/usr/bin/env bash
# Start the web application.   scripts/run_server.sh [port]   (default 8000)
source "$(dirname "$0")/common.sh"
require_venv
PORT="${1:-8000}"
cd drowsiness_detection_webapp
"$PY" manage.py migrate --noinput >/dev/null
info "Open http://127.0.0.1:$PORT/  (webcam needs localhost or HTTPS). Press Ctrl+C to stop."
exec "$PY" manage.py runserver "127.0.0.1:$PORT"
