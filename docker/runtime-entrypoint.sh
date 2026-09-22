#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Entrypoint for the fni8-serve runtime image. Two ways to run:
#   1. Pass CLI flags directly — forwarded verbatim to the server:
#        docker run ... ghcr.io/jajmangold/fni8-serve --model /models/m.fni8 --tokenizer ...
#   2. Pass NO args — the invocation is built from env vars (mount your own .fni8):
#        docker run ... -e FNI8_MODEL=/models/m.fni8 -e FNI8_TOKENIZER=/models/tok ...
set -euo pipefail

if [ "$#" -eq 0 ]; then
  : "${FNI8_MODEL:?set FNI8_MODEL=/models/your-model.fni8 (or pass server CLI flags)}"
  set -- --model "$FNI8_MODEL" \
         --host "${FNI8_HOST:-0.0.0.0}" --port "${FNI8_PORT:-8000}"
  [ -n "${FNI8_TOKENIZER:-}" ]           && set -- "$@" --tokenizer "$FNI8_TOKENIZER"
  [ -n "${FNI8_SERVED_MODEL_NAME:-}" ]   && set -- "$@" --served-model-name "$FNI8_SERVED_MODEL_NAME"
  [ -n "${FNI8_MAX_LEN:-}" ]             && set -- "$@" --max-len "$FNI8_MAX_LEN"
  [ -n "${FNI8_MAX_NUM_SEQS:-}" ]        && set -- "$@" --max-num-seqs "$FNI8_MAX_NUM_SEQS"
fi

exec python3 -m fni8serve.api.server "$@"
