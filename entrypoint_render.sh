#!/usr/bin/env bash

# Entrypoint for Render Web Service.
# 1) Apply DB migrations
# 2) Start uvicorn on the Render-provided PORT
set -euo pipefail

echo "=== Render startup: migrations ==="
alembic upgrade head

echo "=== Render startup: uvicorn ==="
exec python -m uvicorn src.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips="*"

