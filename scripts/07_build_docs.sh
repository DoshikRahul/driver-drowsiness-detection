#!/usr/bin/env bash
# Step 7: regenerate README.docx and EXECUTION_GUIDE.docx from the latest results.
# If LibreOffice is installed, PDF copies are also written to docs/build/ for checking.
source "$(dirname "$0")/common.sh"
require_venv
require_python_module docx "Run scripts/00_setup.sh --docs"
require_file runs/summary.json "Run scripts/06_summarize.sh first."
run_step "Step 7: build documents" build_docs.log "$PY" docs/make_docs.py

SOFFICE="$(command -v soffice || command -v libreoffice || true)"
if [[ -n "$SOFFICE" ]]; then
    mkdir -p docs/build
    "$SOFFICE" --headless --convert-to pdf --outdir docs/build README.docx EXECUTION_GUIDE.docx >/dev/null 2>&1 \
        && ok "PDF previews: docs/build/README.pdf, docs/build/EXECUTION_GUIDE.pdf" \
        || warn "LibreOffice PDF conversion failed (the .docx files are fine)."
fi
