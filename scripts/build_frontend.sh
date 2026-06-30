#!/usr/bin/env bash
# scripts/build_frontend.sh
# Build the React frontend into viki/server/static/, which FastAPI serves.
#
# server/static/ is a build artifact (gitignored). The viki/ tree is bind-mounted
# into the container, so building on the host is enough — no Docker rebuild needed.
# Run this once after cloning and after any frontend change you want in `docker compose up`.
#
# Dev loop (no build needed): cd viki/frontend && npm run dev  -> http://localhost:5173
# (proxies /api and the skeleton WebSocket to the FastAPI server on :8000).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../viki/frontend"

cd "$FRONTEND_DIR"

if [ ! -d node_modules ]; then
  echo "[build_frontend] installing dependencies..."
  npm ci
fi

echo "[build_frontend] building -> viki/server/static/ ..."
npm run build

echo "[build_frontend] done. Start the server with: docker compose up"
