from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import torch.nn.functional as F

import triton
import triton.language as tl

from functools import partial

# Constants
LinearNoBias = partial(nn.Linear, bias=False)


# Triton kernels for forward pass
@triton.jit
def _castle_attn_fwd_kernel(
    Q, K, V, QU, KU, VU,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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
        qk = tl.dot(q, tl.trans(k))  # [BM, BK]

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
            t1 = tl.dot(q, tl.trans(v_j))  # [BM, BJ]

            # mask term1 so that positions with j > i are zeroed (keep j <= i)
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            # lookahead matrix = sigmoid(qu_k · ku_j)
            la = tl.dot(qu_k, tl.trans(ku_j))  # [BK, BJ]
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
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= k_ids[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))

        # online softmax update
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i_new[:, None])  # [BM, BK]
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None] + tl.dot(p, v_k)
        l_i = l_i * alpha + l_ij
        m_i = m_i_new

    # finalize
    out = acc / l_i[:, None]

    # store
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, out, mask=offs_m[:, None] < M)


# Triton kernels for backward pass (two-phase recompute)
@triton.jit
def _castle_attn_bwd_kernel(
    Q, K, V, QU, KU, VU,
    dOut,
    dQ, dK, dV, dQU, dKU, dVU,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Castle attention backward kernel.

    Recomputes scores and softmax normalization in two passes over key blocks:
    - Pass 1: online softmax to derive row-wise max m_i and denom l_i.
    - Pass 2: compute probabilities p, then accumulate gradients for
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

    # Pass 1: compute m_i and l_i via online softmax over all key blocks
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

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
        qk = tl.dot(q, tl.trans(k))  # [BM, BK]

        # accumulate Su over all j-blocks
        su_acc = tl.zeros_like(qk)

        for start_j in range(0, N, BLOCK_N):
            j_ids = start_j + offs_n
            v_j_ptrs = vu_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            ku_j_ptrs = ku_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk

            v_j = tl.load(v_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            ku_j = tl.load(ku_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            valid_j = j_ids < N

            # t1(i,j) = q_i · vu_j (masked j <= i)
            t1 = tl.dot(q, tl.trans(v_j))  # [BM, BJ]
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            # la(k,j) = sigmoid(qu_k · ku_j) (masked j > k)
            la = tl.dot(qu_k, tl.trans(ku_j))  # [BK, BJ]
            la = tl.sigmoid(la)
            mask_la = (j_ids[None, :] > k_ids[:, None]) & valid_j[None, :] & valid_k[:, None]
            la = tl.where(mask_la, la, 0.0)

            # accumulate su over j: [BM,BK]
            su_acc += tl.dot(t1, tl.trans(la))

        su_silu = su_acc * tl.sigmoid(su_acc)
        scores = qk - su_silu

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= k_ids[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))

        # online softmax update for this block
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p_blk = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(p_blk, axis=1)
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + l_ij
        m_i = m_i_new

    # Pass 2: compute grads using p with global m_i and l_i
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
        qk = tl.dot(q, tl.trans(k))  # [BM, BK]

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

        # compute scores and p using global m_i and l_i
        su_silu = su_acc * tl.sigmoid(su_acc)
        scores = qk - su_silu
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= k_ids[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))

        p = tl.exp(scores - m_i[:, None]) / l_i[:, None]

        # dp = do @ v_k^T
        dp = tl.dot(do, tl.trans(v_k))  # [BM, BK]

        # dS through softmax
        psum = tl.sum(p * dp, axis=1)  # [BM]
        dS = p * (dp - psum[:, None])  # [BM, BK]

        # grads from qk = q @ k^T
        dq_acc += tl.dot(dS, k)  # [BM,D]
        dk_blk = tl.dot(tl.trans(dS), q)  # [BK,D]

        # dV (causal aggregation): dv_k = p^T @ do
        dv_blk = tl.dot(tl.trans(p), do)  # [BK,D]

        # dSu via s = qk - silu(Su)
        sig_su = tl.sigmoid(su_acc)
        dSu = -dS * (sig_su * (1.0 + su_acc * (1.0 - sig_su)))  # [BM,BK]

        # Now accumulate lookahead path over j-blocks
        for start_j in range(0, N, BLOCK_N):
            j_ids = start_j + offs_n

            v_j_ptrs = vu_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            ku_j_ptrs = ku_base + j_ids[:, None] * stride_kn + offs_d[None, :] * stride_kk

            v_j = tl.load(v_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            ku_j = tl.load(ku_j_ptrs, mask=j_ids[:, None] < N, other=0.0)
            valid_j = j_ids < N

            # recompute t1 and la with masks
            t1 = tl.dot(q, tl.trans(v_j))  # [BM,BJ]
            mask_t1 = (j_ids[None, :] <= offs_m[:, None]) & valid_j[None, :]
            t1 = tl.where(mask_t1, t1, 0.0)

            la = tl.dot(qu_k, tl.trans(ku_j))  # [BK,BJ]
            la = tl.sigmoid(la)
            mask_la = (j_ids[None, :] > k_ids[:, None]) & valid_j[None, :] & valid_k[:, None]
            la = tl.where(mask_la, la, 0.0)

            # dt1 = dSu @ la, mask to j<=i
            dt1 = tl.dot(dSu, la)  # [BM,BJ]
            dt1 = tl.where(mask_t1, dt1, 0.0)

            # dla = dSu^T @ t1
            dla = tl.dot(tl.trans(dSu), t1)  # [BK,BJ]

            # dA = dla * la * (1 - la)
            dA = dla * la * (1.0 - la)  # [BK,BJ]

            # accumulate grads
            # dQ from t1 path: dQ += dt1 @ v_j
            dq_acc += tl.dot(dt1, v_j)

            # dVU_j: dt1^T @ q
            dvu_j = tl.dot(tl.trans(dt1), q)  # [BJ,D]
            dvu_ptrs = dVU_base + j_ids[:, None] * stride_vn + offs_d[None, :] * stride_vk
            tl.atomic_add(dvu_ptrs, dvu_j, mask=j_ids[:, None] < N)

            # dQU_k accum: dA @ ku_j
            dqu_blk += tl.dot(dA, ku_j)  # [BK,D]

            # dKU_j: dA^T @ qu_k
            dku_j = tl.dot(tl.trans(dA), qu_k)  # [BJ,D]
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


# Custom autograd function
class CastleAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, qu, ku, vu, scale, is_causal=True):
        """
        Forward pass of Castle attention
        
        Args:
            q, k, v: Standard attention tensors [batch, heads, seq_len, dim_head]
            qu, ku, vu: Lookahead attention tensors [batch, heads, seq_len, dim_head]
            scale: Scaling factor for queries
            is_causal: Whether to apply causal masking
        """
        batch, heads, seq_len, dim_head = q.shape
        
        # Apply scaling
        q = q * scale
        qu = qu * scale
        
        # Allocate output
        o = torch.empty_like(q)
        
        # Configure grid
        grid = lambda args: (
            triton.cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )
        
        # Launch kernel
        _castle_attn_fwd_kernel[grid](
            q, k, v, qu, ku, vu, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_M=64, BLOCK_N=64, BLOCK_DMODEL=dim_head,
            IS_CAUSAL=is_causal,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, qu, ku, vu, o)
        ctx.scale = scale
        ctx.is_causal = is_causal
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, qu, ku, vu, o = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        batch, heads, seq_len, dim_head = q.shape

        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dqu = torch.zeros_like(qu)
        dku = torch.zeros_like(ku)
        dvu = torch.zeros_like(vu)

        grid = lambda args: (
            triton.cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )

        _castle_attn_bwd_kernel[grid](
            q, k, v, qu, ku, vu,
            do,
            dq, dk, dv, dqu, dku, dvu,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_M=64, BLOCK_N=64, BLOCK_DMODEL=dim_head,
            IS_CAUSAL=is_causal,
        )

        # Chain rule for pre-scaled q and qu
        dq = dq * scale
        dqu = dqu * scale

        return dq, dk, dv, dqu, dku, dvu, None, None


# Apply function
castle_attention = CastleAttentionFunction.apply


# Module wrapper
class TritonCastleAttention(nn.Module):
    """
    Triton-based Castle attention module with flash attention optimization
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Projections for all 6 components (qu, ku, vu, qc, kc, vc)
        self.to_all_qkv = LinearNoBias(dim, dim_inner * 6)
        self.combine_heads = LinearNoBias(dim_inner, dim)
    
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            is_causal: Whether to apply causal masking
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape
        
        # Project to all qkv components
        qkvs = self.to_all_qkv(x)
        qkvs = qkvs.view(batch, seq_len, 6, self.heads, self.dim_head)
        qkvs = qkvs.permute(2, 0, 3, 1, 4)  # [6, batch, heads, seq_len, dim_head]
        
        qu, ku, vu, qc, kc, vc = qkvs.unbind(0)
        
        # Apply Castle attention with Triton kernel
        out = castle_attention(qc, kc, vc, qu, ku, vu, self.scale, True)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous()  # [batch, seq_len, heads, dim_head]
        out = out.view(batch, seq_len, self.heads * self.dim_head)
        
        # Final projection
        out = self.combine_heads(out)
        
        return out
