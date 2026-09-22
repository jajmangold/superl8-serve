#!/usr/bin/env bash
# Quantize a model to .fni8 on **Hugging Face Jobs** — HF's own infrastructure —
# instead of this box. The source model already lives on HF, so the download is
# intra-datacenter (near-instant vs a slow local link), the pure-torch quant runs on
# a CPU flavor (no GPU needed — see the CPU-importable fni8), and the .fni8 output is
# pushed straight to jajmangold/<name>-fni8. Zero local load; submit and forget.
#
# Requires: `hf` CLI logged in (or HF_TOKEN in env) AND pre-paid Jobs credits on the
# account (Settings -> Billing). Cost ~$1-2/model on cpu-performance.
#
#   tools/hf_quant.sh Qwen/Qwen3.6-27B-FP8            # llm, int8
#   tools/hf_quant.sh Qwen/Qwen3.6-27B-FP8 4          # int4
#   tools/hf_quant.sh black-forest-labs/FLUX.1-dev 8 dit
#   FLAVOR=cpu-xl tools/hf_quant.sh zai-org/GLM-5.2-FP8
set -euo pipefail

repo="${1:?usage: hf_quant.sh <repo> [bits] [kind] }"
bits="${2:-8}"
kind="${3:-llm}"
flavor="${FLAVOR:-cpu-performance}"          # 32 vCPU / 256 GB / 1 TB — fits the giants
image="${IMAGE:-python:3.12}"
name="${repo//\//__}"
kindarg=""; [ "$kind" = "dit" ] && kindarg="--kind dit"

hf jobs run --flavor "$flavor" --secrets HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}" \
  --timeout "${TIMEOUT:-6h}" --label "fni8-quant=${repo//\//_}" -d "$image" \
  bash -c "set -euo pipefail
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q safetensors huggingface_hub hf_transfer numpy
    git clone --depth 1 https://github.com/jajmangold/fni8 /fni8
    git clone --depth 1 https://github.com/jajmangold/fni8-serve /serve
    export PYTHONPATH=/fni8:/serve FORGE_BASE=/data HF_HOME=/data/hf \
           HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 FORGE_MIN_FREE_GB=10
    python /serve/tools/forge.py one '$repo' --bits $bits $kindarg
    python /serve/tools/forge.py publish '$repo'"    # repo id is a substring of the parsed parent
