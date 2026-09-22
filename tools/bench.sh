#!/usr/bin/env bash
# Run tools/bench_llm.py inside the fni8 container's `bench` service.
# Before running: check `nvidia-smi` on the host for a free GPU and pass
# `--gpu-load-caveat "..."` through to bench_llm.py.
set -euo pipefail

: "${FNI8_DIR:?Set FNI8_DIR to the fni8 kernel repo path}"
: "${SERVE_DIR:?Set SERVE_DIR to this repo's path}"
: "${ARCHIVE:?Set ARCHIVE to the archive disk mount point}"
BENCH_SERVICE="${BENCH_SERVICE:-bench}"

docker compose -f "$FNI8_DIR/docker-compose.yml" run --rm \
  -v "$ARCHIVE":"$ARCHIVE" \
  -v "$SERVE_DIR":/serve \
  -e FORGE_BASE="${FORGE_BASE:?Set FORGE_BASE to the forge base dir}" \
  "$BENCH_SERVICE" bash -lc "python3 /serve/tools/bench_llm.py $* --out-dir /serve/bench"
