# Multi-GPU MoE with compressed transport

## Architecture

Each GPU owns a contiguous block of layers. MoE layers route tokens to expert-owning
GPUs via the existing `send()`/`recv()` transport seam with compressed activations.

```
GPU 0: layers 0–15, experts 0–15   GPU 1: layers 16–31, experts 16–31
  │                                    │
  ├─ staging_buf[0]                    ├─ staging_buf[16]
  ├─ layer 0 (dense)                   ├─ layer 16 (dense)
  ├─ ...                               ├─ ...
  ├─ layer 8 (MoE) ──────────────────→ ├─ layer 24 (MoE) ←──────────────
  │   router → expert_assignments       │   router → expert_assignments
  │   local experts process locally     │   local experts process locally
  │   remote tokens compressed + sent   │   remote tokens compressed + sent
  │←────────────────────────────────────┤
  ├─ ...                               ├─ ...
  ├─ staging_buf[15]                   ├─ staging_buf[31]
  └─ send to GPU 1 ──────────────────→ └─ recv from GPU 0
                                         └─ layer 32+ or lm_head
```

## Key design decisions

### 1. Routing happens on the source GPU

The router runs on the GPU that owns the MoE layer. It computes expert assignments,
sorts tokens by destination GPU, compresses remote tokens, and sends them. The
receiving GPU decompresses and processes its local experts.

### 2. Transport compression at every PP boundary

Every activation crossing a GPU boundary goes through `superl8.compress_activation()`.
The wire always carries compressed codes (int8/int4/nf4 + fp32 scales), never fp16.
Codec selection per boundary via `select_wire_scheme()`.

### 3. Skip-empty across GPUs

If GPU 1 has no tokens waiting at its staging buffer (all tokens terminated or
routed elsewhere), GPU 1 skips its entire layer range. The staging buffer's
`active_count` is the signal — it crosses the wire as a single int32.

### 4. Expert weight locality

Expert weights stay on the GPU that owns them. Only activations (compressed)
cross the wire. This matches the fleet's bandwidth ratio: HBM (829 GB/s) for
weight reads, PCIe (250 MB/s) for compressed activations.

### 5. Overlap compute with transfer

While GPU 0 processes its local experts, it simultaneously sends remote tokens
to GPU 1. The `send()` function already uses dedicated CUDA streams for
payload/scale transfers. The receiving GPU processes as soon as tokens arrive.

## MoE layer flow (detailed)

```
1. Router (GPU 0):
   logits = x @ gate_weight.T  # [batch, num_experts]
   topk_logits, topk_idx = topk(logits, k)
   expert_mask = topk_idx  # which experts each token is assigned to

2. Sort by destination GPU:
   local_mask = expert_mask < num_local_experts  # experts 0–15
   remote_mask = ~local_mask                       # experts 16–31
   local_tokens = x[local_mask]                    # stay on GPU 0
   remote_tokens = x[remote_mask]                  # go to GPU 1

3. Process local experts (GPU 0):
   for e in active_local_experts:
       tokens_e = local_tokens[expert_mask[local_mask] == e]
       output_e = expert_weights[e](tokens_e)

4. Send remote tokens (GPU 0 → GPU 1):
   handle = send(remote_tokens, dst=1, scheme="int4")  # compressed
   # Meanwhile, GPU 0 continues processing local experts

5. Receive + process (GPU 1):
   remote_tokens = recv(handle)  # decompress on GPU 1
   for e in active_remote_experts:
       tokens_e = remote_tokens[expert_mask[remote_mask] == e]
       output_e = expert_weights[e](tokens_e)
   # Send results back to GPU 0
   result_handle = send(output_e, dst=0, scheme="int4")

6. Combine (GPU 0):
   remote_output = recv(result_handle)  # decompress on GPU 0
   output[local_mask] = local_output
   output[remote_mask] = remote_output
```

## Wire budget analysis

For 9B model, 2-way MoE PP, batch=128, hidden=3584:

| Transfer | Uncompressed | Int4 compressed | With entropy |
|----------|-------------|-----------------|--------------|
| Activation per token | 7 KB | 1.8 KB | 1.2 KB |
| 128 tokens | 896 KB | 230 KB | 150 KB |
| Wire time (250 MB/s) | 3.6 ms | 0.9 ms | 0.6 ms |
| % of layer compute (40 ms) | 9% | 2.3% | 1.5% |

With int4 + entropy: wire is <2% of compute. Effectively free.

## Implementation

### Phase A: MoE-aware staging buffers

Extend `StagingBuffer` to support split routing at MoE layers:
- `staging_buf.split_for_moe(router_weights, expert_map)` → `(local_buf, remote_handle)`
- `staging_buf.merge_from_remote(handle)` → combine local + remote results

### Phase B: Compressed MoE transport

Wire the `send()`/`recv()` seam into the MoE forward pass:
- `WeightStationaryMoE` gains a `transport_dst` parameter (which GPU to send remote tokens to)
- After routing: compress + send remote tokens, process local experts, receive results

### Phase C: Multi-GPU orchestration

Extend `GraphedDecodeLayers` to handle cross-GPU staging:
- Each GPU runs its own per-layer graph loop
- At MoE boundaries, staging buffers are filled by remote sends
- Skip-empty: if no tokens arrive from the remote GPU, skip the entire layer range

### Phase D: Overlap scheduling

Use CUDA streams to overlap:
- Local expert processing on stream A
- Remote token transfer on stream B
- Result receive on stream C

## Validation

- Bit-identical output vs. single-GPU MoE (all 680 tests pass on single-GPU)
- Wire utilization measurement (target: <5% of compute)
- Multi-GPU throughput scaling (target: >1.8× for 2-way PP on MoE models)
- Compression quality: SQNR/cosine per boundary (must pass accuracy gate)
- Skip-empty: measure idle GPU time reduction
