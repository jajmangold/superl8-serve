# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5/3.6 backbone, MiniMax
lightning) — Track-1 fp16 port.

Recurrence (Gated DeltaNet, Yang et al. ICLR'25), per head:

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

with q,k L2-normalized per head, a short causal depthwise conv1d(k=4)+SiLU token
shift on q/k/v, a scalar decay gate alpha_t = exp(-softplus(dt)*exp(A_log)) and a
write gate beta_t = sigmoid(b_t), then a gated RMSNorm on the output. The linear
projections run on the superl8 dp4a GEMM (LinearW8A8); numerically load-bearing
recurrence math stays fp32, whether it uses the eager oracle or exact chunk kernel.

This module is the CORRECT fp path (the numeric oracle). It uses the O(L) scalar
recurrence — unambiguous and exact — with an optional dispatch to superl8's exact
fp32 gated chunk kernel for CUDA prefill. The separate int8 dp4a chunk form is
an approximation for gated models and remains an explicit fallback/opt-in only.
"""

from __future__ import annotations

import os

import superl8
import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _expand_delta_k_heads(x: torch.Tensor, num_v_heads: int, *, tiled: bool) -> torch.Tensor:
    """Broadcast K/Q heads into grouped HF order or llama.cpp's tiled V-head order."""
    rep = num_v_heads // x.shape[1]
    return x.repeat(1, rep, *([1] * (x.dim() - 2))) if tiled else x.repeat_interleave(rep, dim=1)


def recurrent_gated_delta_rule(q, k, v, beta, g, state=None, return_traj=False):
    """Gated delta rule, scalar form. q,k: [B,H,L,Dk] (pre-L2-normed), v: [B,H,L,Dv],
    beta: [B,H,L] in (0,1], g: [B,H,L] = log(alpha) (<=0). `state`: optional
    [B,H,Dv,Dk] fp32 initial S (carried in from a prior call for decode); defaults
    to zero (prefill / stateless). Returns (o [B,H,L,Dv], state_final [B,H,Dv,Dk])
    so a caller can carry `S` across chunked/decode calls instead of losing it.

    `return_traj` (spec-decode verify, task d): also return `state_traj`
    [B,L,H,Dv,Dk] = the state S AFTER each token t, so the caller can commit the
    state after any prefix length without re-running the recurrence."""
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    S = (
        state
        if state is not None
        else torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=q.device)
    )
    alpha = g.exp().float()
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    out = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)
    traj = torch.empty(B, L, H, Dv, Dk, dtype=torch.float32, device=q.device) if return_traj else None
    for t in range(L):
        kt, vt, qt = kf[:, :, t], vf[:, :, t], qf[:, :, t]  # [B,H,D]
        at = alpha[:, :, t][..., None, None]  # [B,H,1,1]
        bt = bf[:, :, t][..., None]  # [B,H,1]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt)  # S_{t-1} k_t
        erase = at * (bt * Sk)[..., None] * kt[..., None, :]
        write = (bt * vt)[..., None] * kt[..., None, :]
        S = at * S - erase + write
        out[:, :, t] = torch.einsum("bhvk,bhk->bhv", S, qt)  # o_t = S_t q_t
        if return_traj:
            traj[:, t] = S
    if return_traj:
        return out.to(v.dtype), S, traj
    return out.to(v.dtype), S


# Kill-switch for MiniMax lightning attention int8 dp4a kernel. Default on;
# auto-degrades to the fp32 scalar reference if the installed superl8 predates the
# kernel. The kernel accelerates prefill (L>1); decode (L==1) always uses the
# fp32 reference (no graph-capturable lightning decode kernel — COVERAGE.md).
_LTN_INT8 = os.environ.get("SUPERL8_LTN_INT8", "1") != "0" and hasattr(
    superl8, "lightning_attn_int8_fwd"
)

# Kill-switch (default on); auto-degrades to the eager reference if the installed
# superl8 predates the decode kernel, so this stays correct on an older prebuilt superl8.
_DND_DECODE = os.environ.get("SUPERL8_DND_DECODE", "1") != "0" and hasattr(
    superl8, "deltanet_recurrent_decode"
)

# Fused per-token DECODE elementwise kernels (same kill-switch / older-superl8
# auto-degrade pattern as _DND_DECODE): the causal conv1d+SiLU token shift and
# the gated output RMSNorm. Each collapses ~4-6 tiny eager ops into ONE launch
# and is CUDA-graph-capturable (no cudaFuncSetAttribute), so they replay inside
# the engine's graphed decode. Fall back to eager if the installed superl8 predates
# them (prebuilt-image compatibility) or SUPERL8_DND_FUSED=0.
_DND_FUSED = os.environ.get("SUPERL8_DND_FUSED", "1") != "0"
_DND_CONV = _DND_FUSED and hasattr(superl8, "causal_conv1d_silu_decode")
_DND_GNORM = _DND_FUSED and hasattr(superl8, "gated_rmsnorm_decode")
# ONE-launch fully-fused DeltaNet decode step: L2-norm(q,k) + GQA expand +
# sigmoid(beta) + g=-softplus(dt+dt_bias)*exp(A_log) + delta-rule recurrence +
# gated output RMSNorm (silu(z)*o), collapsing the ~15-19 tiny glue ops the
# eager chain runs per layer into a single graph-capturable kernel. Its OWN
# kill-switch (SUPERL8_DND_STEP=0) on top of the shared SUPERL8_DND_FUSED, so the
# per-token step fusion can be A/B'd independently of the conv/gnorm fusions;
# also auto-degrades on an superl8 that predates the op.
_DND_STEP = (
    os.environ.get("SUPERL8_DND_STEP", "1") != "0"
    and _DND_FUSED
    and hasattr(superl8, "deltanet_fused_decode")
)

# Fused RAW causal-conv1d DECODE for the LFM2 ShortConv (issue #371): the LFM
# short-conv runs its causal depthwise conv raw (the double gate is applied
# AFTER, in `forward`), so the no-SiLU sibling `superl8.causal_conv1d_decode`
# belongs here — not the SiLU variant DeltaNet uses. Its tail layout
# [B, D, K-1] already matches ShortConv's channel-major per-slot cache, so the
# cache tensor is handed to the kernel AS-IS (no transpose/copy). Own kill-
# switch (SUPERL8_LFM_CONV=0) on top of the hasattr gate; auto-degrades to eager
# if the installed superl8 predates the op (prebuilt-image compatibility), same
# pattern as _DND_DECODE/_DND_CONV.
_LFM_CONV = os.environ.get("SUPERL8_LFM_CONV", "1") != "0" and hasattr(
    superl8, "causal_conv1d_decode"
)

# Exact fp32 gated chunked prefill for Gated DeltaNet (L>1). This is the
# correctness-preserving path: superl8 implements the full alpha/beta-gated
# recurrence with a WY/UT chunk decomposition. It defaults on when available
# and auto-degrades on older prebuilt superl8 images. Set SUPERL8_DND_PREFILL_GATED=0
# for an immediate eager-reference rollback.
_DND_PREFILL_GATED = os.environ.get("SUPERL8_DND_PREFILL_GATED", "1") != "0" and hasattr(
    superl8, "deltanet_gated_chunk_fwd"
)

# Approximate int8 chunked prefill for Gated DeltaNet (L>1). DEFAULT OFF — the
# kernel is ungated (S_t = S_{t-1} + v_t k_t^T); for GATED delta nets
# (Qwen3-Next/3.5/3.6) the erase term is missing, so this can shift greedy
# argmax. Opt in with SUPERL8_DND_PREFILL_INT8=1 only when that approximation is
# acceptable; auto-degrades if the installed superl8 predates the chunk kernel.
_DND_PREFILL_INT8 = os.environ.get("SUPERL8_DND_PREFILL_INT8", "0") == "1" and hasattr(
    superl8, "deltanet_chunk_int8_fwd"
)


def _gated_delta_rule(q, k, v, beta, g, state, L):
    """Dispatch the gated delta rule. For the single-token decode step (L==1, CUDA,
    Dk/Dv<=128) use the fused `superl8.deltanet_recurrent_decode` kernel: same math as
    the eager reference above (validated bit-close against it, cos 1.0), but it
    collapses the ~5 tiny eager ops into ONE launch AND is CUDA-graph-capturable
    (register state, no cudaFuncSetAttribute), so it replays inside the engine's
    graphed decode instead of forcing an eager fallback. For the multi-token prefill
    step (L>1, CUDA, Dk/Dv<=128), first use `superl8.deltanet_gated_chunk_fwd` when
    available. This is the exact fp32 gated WY/UT chunk form: alpha and beta are
    passed directly and the carried state is preserved. If that op is unavailable
    or disabled, the code falls back to the eager oracle; only then may the
    explicitly opt-in approximate `superl8.deltanet_chunk_int8_fwd` path run.

    Decode returns the readout `o` in fp32 (the kernel's native accumulate dtype).
    The gated RMSNorm in the caller upcasts to fp32 anyway, so downcasting to v.dtype
    here would only be re-upcast one op later — a wasted fp32->fp16->fp32 round-trip
    (nsys: `direct_copy_kernel_cuda` + `float16_copy`, both redundant). Keeping it fp32
    is also strictly more accurate (no fp16 rounding before the norm). The eager branch
    still returns v.dtype; the caller's `.float()` normalizes both to fp32."""
    if _DND_DECODE and L == 1 and q.is_cuda and q.shape[-1] <= 128 and v.shape[-1] <= 128:
        o, state = superl8.deltanet_recurrent_decode(
            q.float(), k.float(), v.float(), g.exp().float(), beta.float(), initial_state=state
        )
        return o, state
    if _DND_PREFILL_GATED and L > 1 and q.is_cuda and q.shape[-1] <= 128 and v.shape[-1] <= 128:
        return superl8.deltanet_gated_chunk_fwd(
            q.float(), k.float(), v.float(), g.exp().float(), beta.float(), initial_state=state
        )
    if _DND_PREFILL_INT8 and L > 1 and q.is_cuda and q.shape[-1] <= 128 and v.shape[-1] <= 128:
        alpha = g.exp().float()                                 # [B,nv,L]
        A = torch.cumprod(alpha.clamp_min(1e-30), dim=-1)       # cumulative decay
        bf = beta.float()                                       # [B,nv,L]
        vf = v.float()                                          # [B,nv,L,Dv]
        v_scaled = (bf.unsqueeze(-1) * vf) / A.unsqueeze(-1).clamp_min(1e-30)
        o, state = superl8.deltanet_chunk_int8_fwd(
            q.float(), k.float(), v_scaled, initial_state=state
        )
        o = o * A.unsqueeze(-1)                                 # [B,nv,L,Dv]
        if state is not None:
            state = state * A[:, :, -1:].unsqueeze(-1)
        return o, state
    return recurrent_gated_delta_rule(q, k, v, beta, g, state=state)


class GatedDeltaNetAttention(nn.Module):
    """Linear-attention block. Projections on dp4a; recurrence fp32. GQA-style:
    `num_v_heads` value/output heads, `num_k_heads` query/key heads (broadcast).

    Weights: qkv_proj (merged Q|K|V), beta_proj + a/dt gate params (A_log, dt_bias),
    conv weight [width, kernel], out_proj, and the gated output RMSNorm gain.
    """

    is_recurrent = True  # carries per-slot decode state via ctx.lin_cache
    spec_capture = True  # records verify-token state trajectory (runner task d)

    def __init__(
        self,
        cfg,
        *,
        qkv_proj: QTensor,
        out_proj: QTensor,
        conv_weight,
        a_log,
        dt_bias,
        beta_proj,
        gate_proj,
        norm_gain,
        num_k_heads,
        num_v_heads,
        key_dim,
        value_dim,
        conv_kernel=4,
        z_proj: QTensor | None = None,
        gate_beta_proj: QTensor | None = None,
    ):
        super().__init__()
        self.nk, self.nv = num_k_heads, num_v_heads
        self.kd, self.vd = key_dim, value_dim
        self.tiled_delta_heads = bool(cfg.extra.get("gguf_tiled_linear_attention", False))
        self.conv_kernel = conv_kernel
        self.qkv_proj = LinearW8A8(qkv_proj)
        self.out_proj = LinearW8A8(out_proj)
        if gate_beta_proj is not None:
            self.gate_beta_proj = LinearW8A8(gate_beta_proj)
        else:
            self.beta_proj = LinearW8A8(beta_proj)
            self.gate_proj = LinearW8A8(gate_proj)
        if z_proj is not None:
            self.z_proj = LinearW8A8(z_proj)
        self.register_buffer("conv_weight", conv_weight, persistent=False)  # [Wc, K]
        self.A_log = nn.Parameter(a_log)
        self.dt_bias = nn.Parameter(dt_bias)
        self.norm = RMSNorm(value_dim, cfg.rms_norm_eps, norm_gain)

    def _project_gate_beta(self, hidden):
        """Project raw dt/gate and beta logits, using one GEMM when weights are fused."""
        if hasattr(self, "gate_beta_proj"):
            return self.gate_beta_proj(hidden).split([self.nv, self.nv], dim=-1)
        return self.gate_proj(hidden), self.beta_proj(hidden)

    def _conv(self, x, tail=None, return_traj=False):
        """Causal depthwise conv1d(k) + SiLU. x: [B, L, Wc]. `tail`: optional
        [B, K-1, Wc] trailing raw (pre-conv) window from the previous call — carries
        decode history in instead of zero-padding, which would forget it every step.
        Returns (activated [B, L, Wc], new_tail [B, K-1, Wc]).

        `return_traj` (spec-decode verify, task d): also return `tail_traj`
        [B, L, K-1, Wc] = the conv tail AFTER each token t, so the caller can commit
        the tail matching any accepted prefix length."""
        B, L, W = x.shape
        K = self.conv_kernel
        if tail is None:
            tail = x.new_zeros(B, K - 1, W)
        # Fused single-launch decode path (L==1, CUDA, K<=8): one causal
        # conv1d+SiLU kernel replaces cat/conv1d/slice/silu and is
        # CUDA-graph-capturable. Same math as the eager branch below (validated
        # bit-close, cos 1.0). Falls back to eager for prefill / CPU / older superl8.
        if _DND_CONV and L == 1 and x.is_cuda and K <= 8 and not return_traj:
            out2d, new_tail = superl8.causal_conv1d_silu_decode(
                x.reshape(B, W), self.conv_weight, tail
            )
            return out2d.reshape(B, 1, W), new_tail
        xt = torch.cat([tail, x], dim=1)  # [B,K-1+L,Wc]
        new_tail = xt[:, -(K - 1) :] if K > 1 else x.new_zeros(B, 0, W)
        y = F.silu(F.conv1d(xt.transpose(1, 2), self.conv_weight.unsqueeze(1), groups=W).transpose(1, 2))
        if return_traj:
            # tail AFTER token t = the K-1 raw inputs ending at t: xt[:, t+1 : t+K].
            KM = K - 1
            tail_traj = (
                torch.stack([xt[:, t + 1 : t + 1 + KM] for t in range(L)], dim=1)
                if KM > 0
                else x.new_zeros(B, L, 0, W)
            )  # [B, L, K-1, Wc]
            return y, new_tail, tail_traj
        return y, new_tail

    def forward(self, hidden, positions, ctx, layer_idx):
        cache = ctx.lin_cache if ctx is not None else None
        # Spec-decode verify (task d): to commit a recurrent state BYTE-IDENTICAL to
        # plain decode, process the S verify tokens as S CONSECUTIVE L=1 decode steps
        # — the exact same fused conv / recurrence / gated-norm kernels non-spec decode
        # runs (the eager L>1 path is only bit-*close*, which flips tie-breaks on some
        # wheels/GPUs). Snapshot the state + conv tail after each token so the runner
        # can commit the state after any accepted prefix length with no re-decode. The
        # batched attention layers still run once over [B,S]; only these recurrent
        # layers loop (their projections are a small fraction of the model).
        if cache is not None and getattr(cache, "capturing_verify", False):
            B, L, _ = hidden.shape
            outs, st_traj, tl_traj = [], [], []
            cache._vcap = False  # the per-token _run calls are ordinary L=1 decode steps
            try:
                for t in range(L):
                    pos_t = positions[:, t : t + 1] if getattr(positions, "dim", lambda: 0)() == 2 else positions
                    outs.append(self._run(hidden[:, t : t + 1], pos_t, ctx, layer_idx))
                    st_traj.append(cache.get_state(layer_idx))  # [B, ...] copy, after token t
                    tl_traj.append(cache.get_conv_tail(layer_idx))
            finally:
                cache._vcap = True
            state_traj = torch.stack(st_traj, dim=1) if st_traj[0] is not None else None
            conv_traj = torch.stack(tl_traj, dim=1) if tl_traj[0] is not None else None
            cache.record_verify_traj(layer_idx, state_traj, conv_traj)
            return torch.cat(outs, dim=1)
        return self._run(hidden, positions, ctx, layer_idx)

    def _fused_decode_step(self, hidden, q, k, v, cache, layer_idx, B):
        """One `superl8.deltanet_fused_decode` launch for the L==1 / 128-128 decode step:
        absorbs L2-norm(q,k), GQA expand, sigmoid(beta), g=-softplus(dt+dt_bias)*
        exp(A_log), the delta-rule recurrence, and the gated output RMSNorm (silu(z)*o).
        Byte-for-byte the same fp32 math as the eager chain in `_run` (its numeric
        oracle, cos >= 0.9999); the win is launch-count / HBM reduction. q_scale =
        1/sqrt(kd) lands on the *normalised* query inside the kernel — a pre-scale would
        be divided straight out by the kernel's internal L2-norm. `q`/`k`/`v` are the
        raw post-conv splits [B, 1, nk*kd] / [B, 1, nv*vd]."""
        state = cache.get_state(layer_idx) if cache is not None else None
        qf = q.reshape(B, self.nk, 1, self.kd).float()
        kf = k.reshape(B, self.nk, 1, self.kd).float()
        vf = v.reshape(B, self.nv, 1, self.vd).float()
        dt, bl = self._project_gate_beta(hidden)
        dt_f = dt.reshape(B, self.nv, 1).float()   # raw dt logits
        bl_f = bl.reshape(B, self.nv, 1).float()   # raw beta logits
        zf = None
        if hasattr(self, "z_proj"):
            zf = self.z_proj(hidden).reshape(B, self.nv, 1, self.vd).float()
        o, state = superl8.deltanet_fused_decode(
            qf, kf, vf, dt_f, bl_f, self.A_log.float(), self.dt_bias.float(),
            self.norm.weight.float(), z=zf, initial_state=state,
            q_scale=float(self.kd) ** -0.5, eps=float(self.norm.eps),
        )
        if cache is not None:
            cache.set_state(layer_idx, state)
        # out is [B, nv, 1, vd] head-major & contiguous → flattens directly to
        # [B, 1, nv*vd], the same layout the eager `o.reshape(B, L, nv*vd)` yields.
        o = o.reshape(B, 1, self.nv * self.vd)
        return self.out_proj(o.to(hidden.dtype))

    def _run(self, hidden, positions, ctx, layer_idx):
        """One DeltaNet forward over L tokens (L==1 → fused decode kernels; L>1 → eager
        prefill recurrence). Reads/updates the per-slot recurrent state + conv tail."""
        B, L, _ = hidden.shape
        cache = ctx.lin_cache if ctx is not None else None
        conv_tail = cache.get_conv_tail(layer_idx) if cache is not None else None
        qkv, conv_tail = self._conv(self.qkv_proj(hidden), conv_tail)
        if cache is not None:
            cache.set_conv_tail(layer_idx, conv_tail)
        qk = self.nk * self.kd
        q, k, v = qkv.split([qk, qk, self.nv * self.vd], dim=-1)
        # Fully-fused decode fast path (L==1, CUDA, Dk==Dv==128): ONE launch for the
        # whole glue. Byte-for-byte the eager chain below, which stays the fallback +
        # oracle; auto-degrades for prefill / CPU / other dims / an older superl8.
        if (
            _DND_STEP
            and not self.tiled_delta_heads
            and L == 1
            and hidden.is_cuda
            and self.kd == 128
            and self.vd == 128
        ):
            return self._fused_decode_step(hidden, q, k, v, cache, layer_idx, B)
        # HF gated-delta-rule scales the (l2-normed) query by 1/sqrt(head_k_dim) before
        # the readout (`query = query * scale`). Applied here (not inside the shared
        # recurrence, which stays a pure delta rule) since it is a per-model readout
        # scale. Omitting it inflates the pre-norm output ~sqrt(Dk)x and — via the gated
        # RMSNorm eps — rotates the normed output (cos ~0.84 vs HF instead of 1.0).
        # Q/K normalization and the recurrence are numerically load-bearing fp32.
        # Cast BEFORE the norm: normalizing the fp16 projection output and only then
        # upcasting changed the normalized vectors enough to flip downstream int8
        # activation codes (the eager oracle missed fused decode by rel-L1 0.00249).
        q = (
            _l2norm(q.view(B, L, self.nk, self.kd).float()) * (self.kd**-0.5)
        ).transpose(1, 2)
        k = _l2norm(k.view(B, L, self.nk, self.kd).float()).transpose(1, 2)
        v = v.view(B, L, self.nv, self.vd).float().transpose(1, 2)
        # broadcast k/q heads to value heads (GQA)
        q = _expand_delta_k_heads(q, self.nv, tiled=self.tiled_delta_heads)
        k = _expand_delta_k_heads(k, self.nv, tiled=self.tiled_delta_heads)
        dt, beta = self._project_gate_beta(hidden)
        beta = torch.sigmoid(beta).transpose(1, 2)  # [B,nv,L]
        beta = beta.reshape(B, self.nv, L) if beta.dim() == 3 else beta
        dt = dt.transpose(1, 2)
        g = -F.softplus(dt.float() + self.dt_bias.view(1, -1, 1)) * self.A_log.exp().view(1, -1, 1)
        state = cache.get_state(layer_idx) if cache is not None else None
        o, state = _gated_delta_rule(
            q, k, v, beta.reshape(B, self.nv, L), g.reshape(B, self.nv, L), state, L
        )
        if cache is not None:
            cache.set_state(layer_idx, state)
        # HF Qwen3_5RMSNormGated ("norm BEFORE gate"): a PER-HEAD RMS over head_v_dim,
        # then the per-head gain, THEN the z gate (silu) — in that order. Done in fp32
        # (the gated norm is numerically load-bearing, never quantized). The gain is
        # per-head (length vd), so normalize over the last (vd) axis, not nv*vd.
        # [B, L, nv, vd] in fp32. `.float()` is a no-op (free) when _gated_delta_rule
        # already returned fp32 for the decode path (#228); it upcasts only the eager
        # (prefill) fp16 output.
        o = o.transpose(1, 2).float()  # [B, L, nv, vd]
        z = None
        if hasattr(self, "z_proj"):
            z = self.z_proj(hidden).view(B, L, self.nv, self.vd).float()
        # Fused single-launch gated RMSNorm (HF Qwen3_5RMSNormGated, "norm before
        # gate") for the L==1 decode step: per-head RMS over vd, ×gain, ×silu(z)
        # in ONE CUDA-graph-capturable kernel replacing ~6 eager fp32 ops. Same
        # math as the eager branch (fp32, load-bearing, never quantized). Falls
        # back to eager for prefill / CPU / older superl8 / vd>128.
        if _DND_GNORM and L == 1 and o.is_cuda and self.vd <= 128:
            o = superl8.gated_rmsnorm_decode(
                o.reshape(B, self.nv, self.vd),
                self.norm.weight.float(),
                z.reshape(B, self.nv, self.vd) if z is not None else None,
                float(self.norm.eps),
            ).reshape(B, L, self.nv, self.vd)
        else:
            o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm.eps)
            o = o * self.norm.weight.float()
            if z is not None:
                # Prefill runs under inference_mode.  Materialising both silu(z)
                # and the product costs one extra [B,L,nv,vd] allocation; on a
                # 16-GiB Qwen3.8 card that 46 MiB temporary materially raises the
                # peak. Keep the autograd path functional, but discard the product
                # temporary in the serving path by mutating the dead recurrence output.
                if torch.is_inference_mode_enabled():
                    o.mul_(F.silu(z))
                else:
                    o = o * F.silu(z)
        o = o.reshape(B, L, self.nv * self.vd)
        return self.out_proj(o.to(hidden.dtype))


def lightning_attention(q, k, v, slopes, state=None):
    """MiniMax lightning attention (TransNormer): data-INDEPENDENT fixed decay, no
    delta correction. S_t = ratio_h S_{t-1} + k_t^T v_t ; o_t = q_t S_t, with per-head
    ratio = exp(-slope). q,k: [B,H,L,Dk], v: [B,H,L,Dv], slopes: [H]. Scalar oracle.
    `state`: optional [B,H,Dv,Dk] fp32 initial S (carried in from a prior call for
    decode); defaults to zero. Returns (o [B,H,L,Dv], state_final [B,H,Dv,Dk])."""
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    S = (
        state
        if state is not None
        else torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=q.device)
    )
    ratio = torch.exp(-slopes.float()).view(1, H, 1, 1)
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)
    for t in range(L):
        S = ratio * S + vf[:, :, t][..., :, None] * kf[:, :, t][..., None, :]  # k^T v outer
        out[:, :, t] = torch.einsum("bhvk,bhk->bhv", S, qf[:, :, t])
    return out.to(v.dtype), S


def _lightning_attn_dispatch(q, k, v, slopes, state=None):
    """Dispatch to ``superl8.lightning_attn_int8_fwd`` for prefill (L>1, CUDA, kernel
    available), falling back to the fp32 scalar reference otherwise (decode L==1,
    older superl8, or env SUPERL8_LTN_INT8=0). Returns ``(o, state_final)`` — the same
    signature as :func:`lightning_attention` so callers are unchanged.

    The int8 kernel implements the UN-GATED (decay-free) recurrence:
    ``S_t = S_{t-1} + v_t k_t^T``. MiniMax ALiBi slopes produce per-head decay
    ratios, so for non-trivial slopes the fp32 reference is always used (the
    reparameterisation ``q'_t = ratio^t q_t, k'_t = ratio^{-t} k_t`` is exact in
    fp32 but underflows/overflows in int8 for real prefill lengths). When slopes
    are all approx 1.0 (decay-free models, or unit tests) the int8 dp4a kernel fires."""
    L = q.shape[2]
    if _LTN_INT8 and L > 1 and q.is_cuda and slopes.abs().max().item() < 1e-6:
        o, state = superl8.lightning_attn_int8_fwd(q.float(), k.float(), v.float(), initial_state=state)
        return o, state
    return lightning_attention(q, k, v, slopes, state=state)


def lightning_slopes(num_heads: int, device="cpu") -> torch.Tensor:
    """ALiBi-style per-head decay slopes: 2^(-8*(h+1)/H)."""
    h = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * h / num_heads)


def _short_conv_segmented(u, conv_weight, cu_seqlens, kernel=3):
    """Segmented causal depthwise conv1d for PACKED ragged ShortConv input.

    `u`: [T, D] real tokens packed in `cu_seqlens` order. `cu_seqlens`: [S+1]
    cumulative token counts per sequence. Inserts `kernel-1` zeros between the
    sequences and `kernel-1` zeros up front (one scatter via ``index_copy_``), so
    a SINGLE ``F.conv1d`` over the padded stream computes each sequence's causal
    conv with no cross-sequence leakage; the real tokens' outputs are then
    gathered. Returns (out [T, D], tails [S, D, K-1]) where tails[i] is the exact
    per-sequence trailing `kernel-1` raw window — BIT-IDENTICAL to running the
    serial ``_conv`` on each sequence separately."""
    T, D = u.shape
    K = kernel
    sep = K - 1
    cu = cu_seqlens.to(device=u.device, dtype=torch.int64)
    S = cu.numel() - 1
    r = torch.arange(T, device=u.device)
    seq_id = torch.searchsorted(cu, r, right=True) - 1  # [T] which packed sequence
    dst = r + seq_id * sep + sep                        # [T] stream position per real token
    stream = u.new_zeros(T + sep * (S + 1), D)
    stream.index_copy_(0, dst, u)                       # scatter K-1 zeros between, prepend K-1
    y_all = F.conv1d(stream.transpose(0, 1).unsqueeze(0), conv_weight, groups=D).squeeze(0).transpose(0, 1)
    out = y_all[dst - sep]                              # [T, D] real-token conv outputs
    i = torch.arange(S, device=u.device)
    end = cu[1:] + i * sep + sep                        # [S] stream index after each seq's tokens
    tail_idx = end[:, None] - sep + torch.arange(sep, device=u.device)  # [S, K-1] stream positions
    tails = stream[tail_idx].permute(0, 2, 1)           # [S, D, K-1] per-seq trailing raw window
    return out, tails


class ShortConv(nn.Module):
    """LFM2 double-gated causal depthwise short conv (LIV): out = out_proj(C * conv(B*x)),
    with (B,C,x) = in_proj(h).chunk(3). in_proj/out_proj on dp4a; conv is depthwise k=3."""

    is_recurrent = True  # carries per-slot decode conv-tail via ctx.lin_cache
    spec_capture = True  # records verify-token conv-tail trajectory (runner task d)
    varlen_prefill_safe = True  # segmented causal conv handles packed cu_seqlens

    def __init__(self, dim: int, *, in_proj: QTensor, out_proj: QTensor, conv_weight, kernel=3):
        super().__init__()
        self.dim = dim
        self.kernel = kernel
        self.in_proj = LinearW8A8(in_proj)
        self.out_proj = LinearW8A8(out_proj)
        self.register_buffer("conv_weight", conv_weight, persistent=False)  # [dim, 1, k]

    def _conv(self, u, tail=None, return_traj=False):
        """Causal depthwise conv1d(k) over the last dimension of `u` [B,D,L]. `tail`:
        optional [B,D,K-1] trailing window from the previous call — carries decode
        history in instead of zero-padding. Returns (activated [B,L,D], new_tail
        [B,D,K-1]). `return_traj`: also return the per-token tail trajectory
        [B, L, D, K-1] (spec-decode verify, task d)."""
        B, D, L = u.shape
        K = self.kernel
        if tail is None:
            tail = u.new_zeros(B, D, K - 1)
        # Fused single-launch DECODE path (L==1, CUDA, K<=8): one raw causal
        # conv1d+tail-roll kernel replaces the eager cat/conv1d/transpose/slice
        # swarm (Nsys attributes 22 generic conv1D_NCHW_general calls, ~21.9% of
        # the B512 graph replay) and is CUDA-graph-capturable. The kernel's tail
        # layout [B, D, K-1] matches ShortConv's cache exactly, so `tail` is
        # handed over without a transpose or copy. Same math as the eager branch
        # below; falls back to eager for prefill / CPU / K>8 / verify
        # (return_traj) / an superl8 that predates the op.
        if _LFM_CONV and L == 1 and u.is_cuda and K <= 8 and not return_traj:
            out2d, new_tail = superl8.causal_conv1d_decode(
                u.reshape(B, D), self.conv_weight.squeeze(1), tail
            )
            return out2d.reshape(B, 1, D), new_tail
        u_ext = torch.cat([tail, u], dim=-1)  # [B,D,K-1+L]
        new_tail = u_ext[:, :, -(K - 1) :] if K > 1 else u.new_zeros(B, D, 0)
        y = F.conv1d(u_ext, self.conv_weight, groups=D).transpose(1, 2)
        if return_traj:
            KM = K - 1
            tail_traj = (
                torch.stack([u_ext[:, :, t + 1 : t + 1 + KM] for t in range(L)], dim=1)
                if KM > 0
                else u.new_zeros(B, L, D, 0)
            )  # [B, L, D, K-1]
            return y, new_tail, tail_traj
        return y, new_tail

    def forward(self, x, positions=None, ctx=None, layer_idx=0):
        B, L, _ = x.shape
        bcx = self.in_proj(x)  # [B,L,3D]
        Bg, Cg, xg = bcx.chunk(3, dim=-1)
        u = (Bg * xg).transpose(1, 2)  # [B,D,L]
        cache = ctx.lin_cache if ctx is not None else None
        cu_seqlens = getattr(ctx, "cu_seqlens", None) if ctx is not None else None
        if cu_seqlens is not None:
            # Packed ragged prefill: ONE segmented causal conv over the packed
            # stream (helper scatters K-1 zeros between/prepended sequences), and
            # bind the exact per-sequence trailing windows on the already-bound
            # slots. `u` is [1, D, T] here (T = packed total tokens) → [T, D].
            y, tails = _short_conv_segmented(
                u.transpose(1, 2).reshape(-1, u.shape[1]), self.conv_weight, cu_seqlens, self.kernel
            )
            if cache is not None:
                cache.set_conv_tail(layer_idx, tails)
            out = self.out_proj(Cg.reshape(-1, Cg.shape[-1]) * y)
            return out.reshape(*x.shape[:-1], self.dim)  # keep packed [1, T, D] rank
        capture = cache is not None and getattr(cache, "capturing_verify", False)
        tail = cache.get_conv_tail(layer_idx) if cache is not None else None
        if capture:
            y, new_tail, tail_traj = self._conv(u, tail, return_traj=True)
            # State-free layer: record only the conv-tail trajectory ([B,L,D,K-1]).
            cache.record_verify_traj(layer_idx, None, tail_traj)
        else:
            y, new_tail = self._conv(u, tail)
        if cache is not None:
            cache.set_conv_tail(layer_idx, new_tail)
        return self.out_proj(Cg * y)
