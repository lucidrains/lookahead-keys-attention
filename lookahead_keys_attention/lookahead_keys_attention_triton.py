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
    """Castle attention forward kernel"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Batch and head offsets
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    off_k = off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    off_v = off_hz * stride_vh + offs_n[None, :] * stride_vn + offs_d[:, None]
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    
    # Load Q for this block
    q = tl.load(Q + off_q, mask=offs_m[:, None] < M, other=0.0)
    qu = tl.load(QU + off_q, mask=offs_m[:, None] < M, other=0.0)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + offs_n
        
        # Load K, V, KU, VU blocks
        k = tl.load(K + off_k, mask=offs_n_curr[:, None] < N, other=0.0)
        v = tl.load(V + off_v, mask=offs_n_curr[:, None] < N, other=0.0)
        ku = tl.load(KU + off_k, mask=offs_n_curr[:, None] < N, other=0.0)
        vu = tl.load(VU + off_v, mask=offs_n_curr[:, None] < N, other=0.0)
        
        # Compute standard attention scores
        qk = tl.dot(q, tl.trans(k))
        
        # Compute lookahead scores
        quku = tl.dot(qu, tl.trans(ku))
        lookahead_attn = tl.sigmoid(quku)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
            lookahead_attn = tl.where(~causal_mask, lookahead_attn, 0.0)
        
        # Compute Su term
        su = tl.dot(lookahead_attn, vu)
        su_silu = tl.libdevice.silu(su)
        
        # Combine scores
        scores = qk - su_silu
        
        # Online softmax
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exponentials
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, axis=1)
        
        # Update accumulator with rescaling
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None] + tl.dot(exp_scores, v)
        l_i = l_i * alpha + l_ij
        m_i = m_i_new
        
        # Update pointers for next block
        off_k += BLOCK_N * stride_kn
        off_v += BLOCK_N * stride_vn
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < M)


# Triton kernels for backward pass
@triton.jit
def _castle_attn_bwd_kernel(
    Q, K, V, QU, KU, VU, Out, dOut,
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
    """Castle attention backward kernel"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Batch and head offsets
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    off_k = off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    off_v = off_hz * stride_vh + offs_n[None, :] * stride_vn + offs_d[:, None]
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    
    # Load Q, QU and gradient for this block
    q = tl.load(Q + off_q, mask=offs_m[:, None] < M, other=0.0)
    qu = tl.load(QU + off_q, mask=offs_m[:, None] < M, other=0.0)
    do = tl.load(dOut + off_o, mask=offs_m[:, None] < M, other=0.0)
    
    # Initialize gradient accumulators
    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dqu_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over K, V blocks for backward pass
    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + offs_n
        
        # Load K, V, KU, VU blocks
        k = tl.load(K + off_k, mask=offs_n_curr[:, None] < N, other=0.0)
        v = tl.load(V + off_v, mask=offs_n_curr[:, None] < N, other=0.0)
        ku = tl.load(KU + off_k, mask=offs_n_curr[:, None] < N, other=0.0)
        vu = tl.load(VU + off_v, mask=offs_n_curr[:, None] < N, other=0.0)
        
        # Recompute forward pass for this block
        qk = tl.dot(q, tl.trans(k))
        quku = tl.dot(qu, tl.trans(ku))
        lookahead_attn = tl.sigmoid(quku)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
            lookahead_attn = tl.where(~causal_mask, lookahead_attn, 0.0)
        
        # Compute Su term
        su = tl.dot(lookahead_attn, vu)
        su_silu = tl.libdevice.silu(su)
        
        # Combine scores  
        scores = qk - su_silu
        
        # Compute attention weights
        p = tl.softmax(scores, axis=1)
        
        # Compute gradient of attention scores
        dp = tl.dot(do, tl.trans(v))
        
        # Gradient through softmax
        softmax_scale = p * (dp - tl.sum(p * dp, axis=1)[:, None])
        
        # Gradient through scores = qk - silu(su)
        dqk = softmax_scale
        dsu_silu = -softmax_scale
        
        # Gradient through silu
        dsu = dsu_silu * (1 + su * tl.sigmoid(su))
        
        # Gradient through su = lookahead_attn @ vu
        dlookahead_attn = tl.dot(dsu, tl.trans(vu))
        
        # Gradient through sigmoid
        dquku = dlookahead_attn * lookahead_attn * (1 - lookahead_attn)
        
        # Accumulate gradients for Q and QU
        dq_acc += tl.dot(dqk, k)
        dqu_acc += tl.dot(dquku, ku)
        
        # Update pointers for next block
        off_k += BLOCK_N * stride_kn
        off_v += BLOCK_N * stride_vn
    
    # Store gradients
    tl.store(dQ + off_q, dq_acc, mask=offs_m[:, None] < M)
    tl.store(dQU + off_q, dqu_acc, mask=offs_m[:, None] < M)


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
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=dim_head,
            IS_CAUSAL=is_causal,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, qu, ku, vu, o)
        ctx.scale = scale
        ctx.is_causal = is_causal
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        """
        Backward pass of Castle attention
        """
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
        
        # Configure grid
        grid = lambda args: (
            triton.cdiv(seq_len, args['BLOCK_M']),
            batch * heads,
        )
        
        # Launch backward kernel
        _castle_attn_bwd_kernel[grid](
            q, k, v, qu, ku, vu, o, do,
            dq, dk, dv, dqu, dku, dvu,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=dim_head,
            IS_CAUSAL=is_causal,
        )
        
        # Apply scale gradient
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
        x: Tensor,
        is_causal: bool = True
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
        out = castle_attention(qc, kc, vc, qu, ku, vu, self.scale, is_causal)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous()  # [batch, seq_len, heads, dim_head]
        out = out.view(batch, seq_len, self.heads * self.dim_head)
        
        # Final projection
        out = self.combine_heads(out)
        
        return out