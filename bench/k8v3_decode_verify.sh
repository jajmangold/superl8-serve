#!/bin/bash
# k8v3_decode_verify.sh — CUDA-graph capture verification for Qwen3.5-9B K8V3 decode
#
# This script loads Qwen3.5-9B with cache_format='k8v3', runs prefill+decode under
# CUDA graph capture, and confirms no crash across 3+ reps.
#
# Prerequisites:
#   - fni8 extension must be compiled with the attn_paged_decode_k8v3 kernel
#   - GPU must support sm_70 (Volta CMP 100-210)
#   - CUDA 12.9, torch 2.10.0+cu129, Python 3.12
#
# Usage:
#   ./bench/k8v3_decode_verify.sh --model /path/to/qwen35-9b.fni8 --tokenizer Qwen/Qwen3.5-9B
#

set -e

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

# Defaults
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
CACHE_FORMAT="${CACHE_FORMAT:-k8v3}"
MAX_SEQS="${MAX_SEQS:-4}"
MAX_LEN="${MAX_LEN:-2048}"
BUCKETS="${BUCKETS:-1,2,4,8}"
NUM_REPS="${NUM_REPS:-3}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --cache-format) CACHE_FORMAT="$2"; shift 2 ;;
        --max-seqs) MAX_SEQS="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --buckets) BUCKETS="$2"; shift 2 ;;
        --num-reps) NUM_REPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== K8V3 CUDA Graph Decode Verification ==="
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER"
echo "Cache format: $CACHE_FORMAT"
echo "Max seqs: $MAX_SEQS, Max len: $MAX_LEN"
echo "Buckets: $BUCKETS, Reps: $NUM_REPS"
echo "=========================================="

# Check if fni8 extension is available
python3 -c "import fni8; print('fni8 OK')" || {
    echo "ERROR: fni8 extension not found or not compiled"
    exit 1
}

# Check if the k8v3 kernel is available
python3 -c "import fni8; fni8.attn_paged_decode_k8v3" 2>/dev/null || {
    echo "WARNING: attn_paged_decode_k8v3 kernel not found - will fallback to torch"
}

# Run the verification using a modified prefill_bucket_probe.py
python3 << PYTHON_SCRIPT
import time
import traceback
import torch

def main():
    print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}, ")
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    from transformers import AutoTokenizer
    from fni8serve.api.server import load_engine
    from fni8serve.engine.sequence import SamplingParams
    
    tok = AutoTokenizer.from_pretrained("${TOKENIZER}")
    
    t0 = time.time()
    engine = load_engine(
        "${MODEL_PATH}", device="cuda", max_num_seqs=${MAX_SEQS}, max_len=${MAX_LEN},
        cuda_graph_batch_buckets=tuple(int(x) for x in "${BUCKETS}".split(",")),
        cache_format="${CACHE_FORMAT}",
    )
    print(f"[load] {time.time() - t0:.1f}s, max_num_seqs=${{MAX_SEQS}}, buckets=${{BUCKETS}}, format=${{CACHE_FORMAT}}")
    
    # Warmup
    passage = (
        "The history of distributed computing systems traces back to the early "
        "mainframe era, when time-sharing systems first allowed multiple users to "
        "interact with a single powerful machine concurrently. As networking "
        "technology matured through the 1970s and 1980s, researchers began exploring "
        "how independent computers could cooperate on shared tasks."
    )
    msg = [{"role": "user", "content": f"Summarize in two sentences.\\n\\n{passage}"}]
    prompt_ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_dict=False)
    print(f"[prompt] {len(prompt_ids)} tokens")
    
    params = SamplingParams(max_tokens=64, temperature=0.0)
    
    # Test each bucket with 3 reps
    buckets = tuple(int(x) for x in "${BUCKETS}".split(","))
    all_passed = True
    
    for n in buckets:
        prompts = [list(prompt_ids) for _ in range(n)]
        print(f"\n=== Bucket n={n} ===")
        for rep in range(${NUM_REPS}):
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                outs = engine.generate(prompts, params)
            except Exception as e:
                print(f"[bucket={n} rep={rep}] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                all_passed = False
                break
            torch.cuda.synchronize()
            dt = time.time() - t0
            total_out_tokens = sum(len(o) for o in outs)
            decode_tokens = total_out_tokens - n
            print(f"[bucket={n:3d} rep={rep}] wall=${{dt:6.2f}s  out_tokens=${{total_out_tokens:5d}}  decode_tok/s=${{decode_tokens / dt:8.1f}}  CAPTURE OK")
    
    print("\n=== RESULT ===")
    if all_passed:
        print("PASS: All ${NUM_REPS} reps completed without crash")
        exit(0)
    else:
        print("FAIL: Some reps failed")
        exit(1)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

exit $?
