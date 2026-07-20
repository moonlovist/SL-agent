#!/usr/bin/env bash
set -euo pipefail

case "${1:-cli}" in
  cli|main)
    echo "==> FORMA CLI pipeline starting..."
    exec python /app/scripts/main.py
    ;;
  web|webui)
    echo "==> FORMA WebUI starting on http://0.0.0.0:${GRADIO_SERVER_PORT:-7860} ..."
    exec python /app/scripts/webui.py
    ;;
  bash|shell)
    exec /bin/bash
    ;;
  *)
    echo "Usage: docker compose run forma-{cli,web} [cli|web|bash]"
    echo "  cli   - run the command-line pipeline"
    echo "  web   - start the Gradio WebUI"
    echo "  bash  - open a shell"
    exit 1
    ;;
esac
