#!/bin/bash
# =============================================================================
# bench/bucket_tuning_9b.sh - Bucket tuning benchmark for Qwen3.5-9B
# =============================================================================
#
# Purpose: Benchmark CUDA-graph bucket configurations for the production
# deployment pattern of 8 replicas of Qwen3.5-9B, each with --parallel 1
# (i.e., max_num_seqs=1, single-sequence-at-a-time per replica).
#
# The issue: DEFAULT_BATCH_BUCKETS=(1,2,4,8,16,32,64,128) wastes capture slots
# and VRAM on batch sizes that never occur under --parallel-1-per-replica.
#
# This script compares:
#   1. CURRENT: DEFAULT_BATCH_BUCKETS=(1,2,4,8,16,32,64,128)
#   2. TUNED:  DEFAULT_BATCH_BUCKETS=(1,)
#
# Usage on a GPU host (V100/CMP):
#   CUDA_VISIBLE_DEVICES=0 bash bench/bucket_tuning_9b.sh \
#       --model /path/to/qwen35-9b.b8.fni8 \
#       --tokenizer Qwen/Qwen3.5-9B
#
# Expected output: A table showing captures/replays for each configuration.
#
# Notes:
#   - This script requires fni8-serve and GPU access
#   - No GPU access in sandbox (CPU-only container) - run on GPU host
#   - Do NOT fabricate benchmark numbers - run this on actual GPU hardware
#   - Results written to /tmp/bucket_tuning_results/
#
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/tmp/bucket_tuning_results"

# Default model and tokenizer
MODEL=""
TOKENIZER="Qwen/Qwen3.5-9B"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model <model> [--tokenizer <tokenizer>]"
            echo ""
            echo "Benchmarks CUDA-graph bucket configurations for Qwen3.5-9B"
            echo ""
            echo "Options:"
            echo "  --model       Path to .fni8 model (required)"
            echo "  --tokenizer   Tokenizer name (default: Qwen/Qwen3.5-9B)"
            echo ""
            echo "Example:"
            echo "  bash bench/bucket_tuning_9b.sh --model /path/to/qwen35-9b.b8.fni8"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print header
echo "========================================"
echo "Qwen3.5-9B Bucket Tuning Benchmark"
echo "========================================"
echo ""
echo "Model: $MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "This script compares CURRENT vs. TUNED bucket configurations"
echo "for the production deployment pattern (--parallel-1-per-replica)."
echo ""
echo "Running..."
echo ""

# Run benchmark with CURRENT buckets
echo "--- Benchmark: CURRENT DEFAULT_BATCH_BUCKETS ---"
echo "Config: (1, 2, 4, 8, 16, 32, 64, 128)"
echo ""

python3 -m bench.prefill_bucket_probe \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --buckets "1,2,4,8,16,32,64,128" \
    --max-tokens 50 \
    --prefill-buckets "32,48,64,96,128,160,192,224,256,384,512,768,1024,1536,2048" \
    --cache-format int8 \
    2>&1 | tee "${RESULTS_DIR}/current_buckets.log"

echo ""
echo "--- Benchmark: TUNED BATCH_BUCKETS ---"
echo "Config: (1,)"
echo ""

python3 -m bench.prefill_bucket_probe \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --buckets "1" \
    --max-tokens 50 \
    --prefill-buckets "32,48,64,96,128,160,192,224,256,384,512,768,1024,1536,2048" \
    --cache-format int8 \
    2>&1 | tee "${RESULTS_DIR}/tuned_buckets.log"

echo ""
echo "========================================"
echo "Benchmark complete. See $RESULTS_DIR for logs."
echo "========================================"
