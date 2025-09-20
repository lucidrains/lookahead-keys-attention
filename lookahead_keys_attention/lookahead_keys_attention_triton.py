from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.autograd import Function
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError("Triton is not installed. Please install triton to use the Triton implementation.")

from functools import partial
from einops.layers.torch import Rearrange

# Triton kernels for forward pass
@triton.jit
def _castle_attn_fwd_kernel(
    Q, K, V, QU, KU, VU,
    Out, Lse,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, seqlen_q_rounded,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Castle attention forward kernel.

    Implements scores = (qc @ kc^T) - silu(Su), with
    Su(i,k) = sum_j (qc_i · v_j) * sigmoid(qu_k · ku_j) with appropriate
    causal masking applied to term1 (j > i masked) and lookahead (j <= k masked).
    Online softmax is used over k to accumulate output.
    """
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    # map pid_hz to (z, h)
    z = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # base pointers for this (z, h)
    q_base = Q + z * stride_qz + h * stride_qh
    k_base = K + z * stride_kz + h * stride_kh
    v_base = V + z * stride_vz + h * stride_vh
    qu_base = QU + z * stride_qz + h * stride_qh
    ku_base = KU + z * stride_kz + h * stride_kh
    vu_base = VU + z * stride_vz + h * stride_vh
    o_base = Out + z * stride_oz + h * stride_oh

    # load q once for the block of queries
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    # initialize accumulators for online softmax
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)

    # iterate over blocks of keys (k)
    for start_k in range(0, N, BLOCK_N):
        k_ids = start_k + offs_n

        # load k, v, and qu for the current k-block
        k_ptrs = k_base + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_k_ptrs = v_base + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
        qu_k_ptrs = qu_base + k_ids[:, None] * stride_qm + offs_d[None, :] * stride_qk

        k = tl.load(k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        v_k = tl.load(v_k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        qu_k = tl.load(qu_k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        valid_k = k_ids < N

        # base qk scores for this k-block
        qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)))  # [BM, BK]

        # accumulate Su over all j-blocks for this k-block
        su_acc = tl.zeros_like(qk)

        for start_j in range(0, N, BLOCK_N):
            j_ids = start_j + offs_n

            # load v_j and ku_j for this j-block
            v_j_ptrs = vu_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            ku_j_ptrs = ku_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk

            v_j = tl.load(v_j_ptrs, mask=j_ids[:, None] < N, other=0.0)  # [BJ, D]
            ku_j = tl.load(ku_j_ptrs, mask=j_ids[:, None] < N, other=0.0)  # [BJ, D]
            valid_j = j_ids < N

            # term1 = (qc_i · v_j) -> use q as qc (already scaled in host)
            t1 = tl.dot(q.to(tl.float32), tl.trans(v_j.to(tl.float32)))  # [BM, BJ]

            # mask term1 so that positions with j > i are zeroed (keep j <= i)
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            # lookahead matrix = sigmoid(qu_k · ku_j)
            la = tl.dot(qu_k.to(tl.float32), tl.trans(ku_j.to(tl.float32)))  # [BK, BJ]
            la = tl.sigmoid(la)

            # mask lookahead so that only strictly upper (j > k) contributes
            mask_la = (j_ids[None, :] > k_ids[:, None]) & valid_j[None, :] & valid_k[:, None]
            la = tl.where(mask_la, la, 0.0)

            # accumulate su: t1 @ la^T over j dimension -> [BM, BK]
            su_acc += tl.dot(t1, tl.trans(la))

        # combine scores for this k-block
        su_silu = su_acc * tl.sigmoid(su_acc)
        scores = qk - su_silu

        # causal mask for attention over k: keep k <= i, set future to -inf
        causal_mask = offs_m[:, None] >= k_ids[None, :]
        scores = tl.where(causal_mask, scores, -float("inf"))

        # online softmax update
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i_new[:, None])  # [BM, BK]
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_k.to(tl.float32))
        l_i = l_i * alpha + l_ij
        m_i = m_i_new

    # finalize
    out = acc / l_i[:, None]

    # compute and store LSE (log-sum-exp)
    lse_i = m_i + tl.log(l_i)
    lse_ptrs = Lse + pid_hz * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < M)

    # store
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, out, mask=offs_m[:, None] < M)


# Preprocess kernel for backward pass (computes delta = do * o)
@triton.jit
def _castle_attn_bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    Z, H, M,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Preprocess kernel to compute delta = sum(do * o, axis=-1)"""
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    # map pid_hz to (z, h)
    z = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # base pointers for this (z, h)
    o_base = Out + z * stride_oz + h * stride_oh
    do_base = DO + z * stride_doz + h * stride_doh

    # load o and do
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok

    o = tl.load(o_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)

    # compute delta = sum(o * do, axis=-1)
    delta = tl.sum(o * do, axis=1)

    # store delta
    delta_ptrs = Delta + pid_hz * M + offs_m
    tl.store(delta_ptrs, delta, mask=offs_m < M)


# Triton kernels for backward pass (using stored LSE)
@triton.jit
def _castle_attn_bwd_kernel(
    Q, K, V, QU, KU, VU,
    dOut, Delta, Lse,
    dQ, dK, dV, dQU, dKU, dVU,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, seqlen_q_rounded,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Castle attention backward kernel.

    Uses stored LSE from forward pass to compute gradients efficiently.
    Only requires a single pass over key blocks, computing probabilities p
    directly from stored LSE, then accumulating gradients for
    Q,K,V (causal) and QU,KU,VU (lookahead) via chain rule.
    """
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    # map pid_hz to (z, h)
    z = pid_hz // H
    h = pid_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # base pointers for this (z, h)
    q_base = Q + z * stride_qz + h * stride_qh
    k_base = K + z * stride_kz + h * stride_kh
    v_base = V + z * stride_vz + h * stride_vh
    qu_base = QU + z * stride_qz + h * stride_qh
    ku_base = KU + z * stride_kz + h * stride_kh
    vu_base = VU + z * stride_vz + h * stride_vh
    do_base = dOut + z * stride_oz + h * stride_oh

    dQ_base = dQ + z * stride_qz + h * stride_qh
    dK_base = dK + z * stride_kz + h * stride_kh
    dV_base = dV + z * stride_vz + h * stride_vh
    dQU_base = dQU + z * stride_qz + h * stride_qh
    dKU_base = dKU + z * stride_kz + h * stride_kh
    dVU_base = dVU + z * stride_vz + h * stride_vh

    # load q and dOut for this query block
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    do_ptrs = do_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    do = tl.load(do_ptrs, mask=offs_m[:, None] < M, other=0.0)

    # load precomputed delta = sum(do * o, axis=-1)
    delta_ptrs = Delta + pid_hz * M + offs_m
    delta = tl.load(delta_ptrs, mask=offs_m < M, other=0.0)

    # load precomputed LSE values from forward pass
    lse_ptrs = Lse + pid_hz * seqlen_q_rounded + offs_m
    lse_i = tl.load(lse_ptrs, mask=offs_m < M, other=0.0)

    # Single pass: compute grads using stored LSE
    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_k in range(0, N, BLOCK_N):
        k_ids = start_k + offs_n

        # load k, v, qu for current k-block
        k_ptrs = k_base + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_k_ptrs = v_base + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
        qu_k_ptrs = qu_base + k_ids[:, None] * stride_qm + offs_d[None, :] * stride_qk

        k = tl.load(k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        v_k = tl.load(v_k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        qu_k = tl.load(qu_k_ptrs, mask=k_ids[:, None] < N, other=0.0)
        valid_k = k_ids < N

        # base qk scores
        qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)))  # [BM, BK]

        # recompute su and also per-j contributions
        su_acc = tl.zeros_like(qk)

        # We will accumulate dQU for this k-block across j_blocks
        dqu_blk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        for start_j in range(0, N, BLOCK_N):
            j_ids = start_j + offs_n
            v_j_ptrs = vu_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            ku_j_ptrs = ku_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk

            v_j = tl.load(v_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            ku_j = tl.load(ku_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            valid_j = j_ids < N

            # t1 and masks
            t1 = tl.dot(q, tl.trans(v_j))  # [BM, BJ]
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            la = tl.dot(qu_k, tl.trans(ku_j))  # [BK, BJ]
            la = tl.sigmoid(la)
            mask_la = (j_ids[None, :] > k_ids[:, None]) & valid_j[None, :] & valid_k[:, None]
            la = tl.where(mask_la, la, 0.0)

            su_acc += tl.dot(t1, tl.trans(la))

        # compute scores and p using stored LSE
        su_silu = su_acc * tl.sigmoid(su_acc)
        scores = qk - su_silu
        causal_mask = offs_m[:, None] >= k_ids[None, :]
        scores = tl.where(causal_mask, scores, -float("inf"))

        p = tl.exp(scores - lse_i[:, None])

        # dp = do @ v_k^T
        dp = tl.dot(do.to(tl.float32), tl.trans(v_k.to(tl.float32)))  # [BM, BK]

        # dS through softmax (using precomputed delta instead of recomputing psum)
        dS = p * (dp - delta[:, None])  # [BM, BK]

        # grads from qk = q @ k^T
        dq_acc += tl.dot(dS, k.to(tl.float32))  # [BM,D]
        dk_blk = tl.dot(tl.trans(dS), q.to(tl.float32))  # [BK,D]

        # dV (causal aggregation): dv_k = p^T @ do
        dv_blk = tl.dot(tl.trans(p), do.to(tl.float32))  # [BK,D]

        # dSu via s = qk - silu(Su)
        # SiLU gradient: d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        sig_su = tl.sigmoid(su_acc)
        dSu = -dS * (sig_su + su_acc * sig_su * (1.0 - sig_su))  # [BM,BK]

        # Now accumulate lookahead path over j-blocks
        for start_j in range(0, N, BLOCK_N):
            j_ids = start_j + offs_n

            v_j_ptrs = vu_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            ku_j_ptrs = ku_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk

            v_j = tl.load(v_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            ku_j = tl.load(ku_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            valid_j = j_ids < N

            # recompute t1 and la with masks
            t1 = tl.dot(q.to(tl.float32), tl.trans(v_j.to(tl.float32)))  # [BM,BJ]
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            la = tl.dot(qu_k.to(tl.float32), tl.trans(ku_j.to(tl.float32)))  # [BK,BJ]
            la = tl.sigmoid(la)
            mask_la = (j_ids[None, :] > k_ids[:, None]) & valid_j[None, :] & valid_k[:, None]
            la = tl.where(mask_la, la, 0.0)

            # dt1 = dSu @ la, mask to j<=i
            dt1 = tl.dot(dSu, la)  # [BM,BJ]
            dt1 = tl.where(mask_t1, dt1, 0.0)

            # dla = dSu^T @ t1
            dla = tl.dot(tl.trans(dSu), t1)  # [BK,BJ]
            # Apply mask to dla (only where j > k)
            dla = tl.where(mask_la, dla, 0.0)

            # dA = dla * la * (1 - la)
            dA = dla * la * (1.0 - la)  # [BK,BJ]

            # accumulate grads
            # dQ from t1 path: dQ += dt1 @ v_j
            dq_acc += tl.dot(dt1, v_j.to(tl.float32))

            # dVU_j: dt1^T @ q
            dvu_j = tl.dot(tl.trans(dt1), q.to(tl.float32))  # [BJ,D]
            dvu_ptrs = dVU_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            tl.atomic_add(dvu_ptrs, dvu_j, mask=j_ids[:, None] < N)

            # dQU_k accum: dA @ ku_j
            dqu_blk += tl.dot(dA, ku_j.to(tl.float32))  # [BK,D]

            # dKU_j: dA^T @ qu_k
            dku_j = tl.dot(tl.trans(dA), qu_k.to(tl.float32))  # [BJ,D]
            dku_ptrs = dKU_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk
            tl.atomic_add(dku_ptrs, dku_j, mask=j_ids[:, None] < N)

        # atomic adds for K, V, QU for current k-block
        dk_ptrs = dK_base + k_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk
        dv_ptrs = dV_base + k_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
        dqu_ptrs = dQU_base + k_ids[:, None] * stride_qm + offs_d[None, :] * stride_qk
        tl.atomic_add(dk_ptrs, dk_blk, mask=k_ids[:, None] < N)
        tl.atomic_add(dv_ptrs, dv_blk, mask=k_ids[:, None] < N)
        tl.atomic_add(dqu_ptrs, dqu_blk, mask=k_ids[:, None] < N)

    # store dQ for this block
    dq_ptrs = dQ_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < M)

# Custom autograd function with Triton
class CastleAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, qu, ku, vu, scale):
        """
        Forward pass of Castle attention

        Args:
            q, k, v: Standard attention tensors [batch, heads, seq_len, dim_head]
            qu, ku, vu: Lookahead attention tensors [batch, heads, seq_len, dim_head]
            scale: Scaling factor for queries
        """
        batch, heads, seq_len, dim_head = q.shape

        # Store original dtype and device
        orig_dtype = q.dtype
        device = q.device

        # Convert to half precision for Triton kernels to reduce shared memory usage
        q = q.half()
        k = k.half()
        v = v.half()
        qu = qu.half()
        ku = ku.half()
        vu = vu.half()

        # Apply scaling
        q = q * scale
        qu = qu * scale

        # Allocate output in half precision
        o = torch.empty_like(q)

        # Allocate LSE tensor for storing log-sum-exp values
        from math import ceil
        seqlen_q_rounded = ceil(seq_len / 128) * 128
        lse = torch.empty((batch, heads, seqlen_q_rounded), device=device, dtype=torch.float32)

        # Configure grid
        def cdiv(a, b):
            return (a + b - 1) // b

        grid = lambda args: (
            cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )
        
        # Dynamic block size selection based on dim_head to avoid shared memory issues
        # Note: Triton dot requires minimum 16x16x16, so we can't go below 16
        # Using half precision should halve the memory usage
        if dim_head <= 32:
            block_m, block_n = 32, 32
        elif dim_head <= 48:
            block_m, block_n = 32, 32
        else:  # dim_head >= 64
            block_m, block_n = 16, 16

        # Launch kernel
        _castle_attn_fwd_kernel[grid](
            q, k, v, qu, ku, vu, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seq_len, seq_len, seqlen_q_rounded,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DMODEL=dim_head,
        )

        # Convert output back to original dtype
        o = o.to(orig_dtype)

        # Save for backward (keep in half precision to save memory)
        ctx.save_for_backward(q, k, v, qu, ku, vu, o.half(), lse)
        ctx.scale = scale
        ctx.orig_dtype = orig_dtype
        ctx.seqlen_q_rounded = seqlen_q_rounded

        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, qu, ku, vu, o, lse = ctx.saved_tensors
        scale = ctx.scale
        orig_dtype = ctx.orig_dtype
        seqlen_q_rounded = ctx.seqlen_q_rounded

        # Convert gradient input to half precision
        do = do.half()

        batch, heads, seq_len, dim_head = q.shape

        # Allocate delta tensor for precomputed do * o
        delta = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

        # Run preprocess kernel to compute delta = sum(do * o, axis=-1)
        def cdiv(a, b):
            return (a + b - 1) // b

        # Dynamic block size selection for preprocess
        if dim_head <= 32:
            block_m = 32
        elif dim_head <= 48:
            block_m = 32
        else:  # dim_head >= 64
            block_m = 16

        grid = lambda args: (
            cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )

        _castle_attn_bwd_preprocess_do_o_dot[grid](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            batch, heads, seq_len, dim_head,
            BLOCK_M=block_m, BLOCK_HEADDIM=dim_head,
        )

        # Allocate gradients in half precision
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dqu = torch.zeros_like(qu, dtype=torch.float32)
        dku = torch.zeros_like(ku)
        dvu = torch.zeros_like(vu)

        # Dynamic block size selection based on dim_head to avoid shared memory issues
        # Note: Triton dot requires minimum 16x16x16, so we can't go below 16
        # Using half precision should halve the memory usage
        if dim_head <= 32:
            block_m, block_n = 32, 32
        elif dim_head <= 48:
            block_m, block_n = 32, 32
        else:  # dim_head >= 64
            block_m, block_n = 16, 16

        grid = lambda args: (
            cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )

        _castle_attn_bwd_kernel[grid](
            q, k, v, qu, ku, vu,
            do, delta, lse,
            dq, dk, dv, dqu, dku, dvu,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seq_len, seq_len, seqlen_q_rounded,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DMODEL=dim_head,
        )

        # Chain rule for pre-scaled q and qu
        dq = dq * scale
        dqu = dqu * scale

        # Convert gradients back to original dtype
        dq = dq.to(orig_dtype)
        dk = dk.to(orig_dtype)
        dv = dv.to(orig_dtype)
        dqu = dqu.to(orig_dtype)
        dku = dku.to(orig_dtype)
        dvu = dvu.to(orig_dtype)

        return dq, dk, dv, dqu, dku, dvu, None

# Apply function
castle_attention_triton = CastleAttentionFunction.apply
