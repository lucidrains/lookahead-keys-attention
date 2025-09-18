from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.autograd import Function
import torch.nn.functional as F

from functools import partial
from einops.layers.torch import Rearrange

# Constants
LinearNoBias = partial(nn.Linear, bias=False)


class CastleAttentionFunction(Function):
    """Pure PyTorch implementation of Castle attention with correct gradients"""
    
    @staticmethod
    def forward(ctx, qc, kc, vc, qu, ku, vu, scale):
        """
        Forward pass following the naive implementation exactly
        """
        batch, heads, seq_len, dim_head = qc.shape
        device = qc.device
        
        # Scale queries
        qu_scaled = qu * scale
        qc_scaled = qc * scale
        
        # Causal mask
        causal_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).triu(1)
        
        # Compute term1 = qc_scaled @ vu^T (masked for j <= i)
        term1 = einsum('bhid,bhjd->bhij', qc_scaled, vu)
        term1 = term1.masked_fill(causal_mask, 0.)
        
        # Compute lookahead = sigmoid(qu_scaled @ ku^T) (masked for j > i)
        lookahead_scores = einsum('bhid,bhjd->bhij', qu_scaled, ku)
        lookahead = lookahead_scores.sigmoid()
        lookahead = lookahead.masked_fill(~causal_mask, 0.)
        
        # Compute Su = term1 @ lookahead^T
        # Note: einsum('bhij,bhkj->bhik', term1, lookahead) treats lookahead as [k,j] indexed
        Su = einsum('bhij,bhkj->bhik', term1, lookahead)
        
        # Compute Sc = qc_scaled @ kc^T
        Sc = einsum('bhid,bhjd->bhij', qc_scaled, kc)
        
        # Compute scores = Sc - silu(Su)
        silu_Su = F.silu(Su)
        scores = Sc - silu_Su
        
        # Apply causal mask
        scores_masked = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn = scores_masked.softmax(dim=-1)
        
        # Output
        out = einsum('bhij,bhjd->bhid', attn, vc)
        
        # Save for backward - save intermediate results we'll need
        ctx.save_for_backward(
            qc_scaled, qu_scaled, ku, vu, kc, vc,
            term1, lookahead, Su, scores, attn
        )
        ctx.scale = scale
        ctx.causal_mask = causal_mask
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass following the gradient formulas from the paper
        """
        (qc_scaled, qu_scaled, ku, vu, kc, vc,
         term1, lookahead, Su, scores, attn) = ctx.saved_tensors
        scale = ctx.scale
        causal_mask = ctx.causal_mask
        
        batch, heads, seq_len, dim_head = qc_scaled.shape
        device = qc_scaled.device
        
        # === Gradient through output ===
        # dvc = attn^T @ grad_out
        dvc = einsum('bhij,bhid->bhjd', attn, grad_out)
        
        # dattn = grad_out @ vc^T
        dattn = einsum('bhid,bhjd->bhij', grad_out, vc)
        
        # === Gradient through softmax ===
        # For each row: d_scores[i] = attn[i] * (dattn[i] - dot(attn[i], dattn[i]))
        dscores_masked = attn * (dattn - (attn * dattn).sum(dim=-1, keepdim=True))
        
        # Through masking (zero gradients where mask is True)
        dscores = dscores_masked.masked_fill(causal_mask, 0.0)
        
        # === Gradient through scores = Sc - silu(Su) ===
        dSc = dscores
        
        # Gradient through silu
        sig_Su = Su.sigmoid()
        dsilu_Su = sig_Su + Su * sig_Su * (1 - sig_Su)
        dSu = -dscores * dsilu_Su
        
        # === Gradient through Sc = qc_scaled @ kc^T ===
        dkc = einsum('bhij,bhid->bhjd', dSc, qc_scaled)
        dqc_from_Sc = einsum('bhij,bhjd->bhid', dSc, kc)
        
        # === Gradient through Su = einsum('bhij,bhkj->bhik', term1, lookahead) ===
        # Su[i,k] = sum_j term1[i,j] * lookahead[k,j]
        # dterm1[i,j] = sum_k dSu[i,k] * lookahead[k,j]
        # dlookahead[k,j] = sum_i term1[i,j] * dSu[i,k]
        
        dterm1 = einsum('bhik,bhkj->bhij', dSu, lookahead)
        dlookahead_kj = einsum('bhij,bhik->bhkj', term1, dSu)
        
        # Apply masks
        dterm1 = dterm1.masked_fill(causal_mask, 0.0)
        # dlookahead is indexed as [k,j] but we need it as [i,j] for the sigmoid gradient
        # Since k and i both range from 0 to seq_len-1, dlookahead_kj can be used as dlookahead_ij
        dlookahead_ij = dlookahead_kj
        dlookahead_ij = dlookahead_ij.masked_fill(~causal_mask, 0.0)
        
        # === Gradient through term1 = qc_scaled @ vu^T ===
        dvu = einsum('bhij,bhid->bhjd', dterm1, qc_scaled)
        dqc_from_term1 = einsum('bhij,bhjd->bhid', dterm1, vu)
        
        # === Gradient through lookahead = sigmoid(qu_scaled @ ku^T) ===
        # Gradient through sigmoid
        sig_deriv = lookahead * (1 - lookahead)
        dlookahead_scores = dlookahead_ij * sig_deriv
        dlookahead_scores = dlookahead_scores.masked_fill(~causal_mask, 0.0)
        
        # Gradient to ku and qu_scaled
        dku = einsum('bhij,bhid->bhjd', dlookahead_scores, qu_scaled)
        dqu_scaled = einsum('bhij,bhjd->bhid', dlookahead_scores, ku)
        
        # === Combine gradients and apply chain rule for scaling ===
        # We computed gradients w.r.t. qc_scaled and qu_scaled
        # Since qc_scaled = qc * scale and qu_scaled = qu * scale,
        # by chain rule: dL/dqc = dL/dqc_scaled * scale
        dqc = (dqc_from_Sc + dqc_from_term1) * scale  
        dqu = dqu_scaled * scale
        
        return dqc, dkc, dvc, dqu, dku, dvu, None


# Apply function
castle_attention = CastleAttentionFunction.apply


# Module wrapper
class TritonCastleAttention(nn.Module):
    """
    CASTLE attention module with corrected PyTorch implementation.
    
    This implementation provides correct gradients for the CASTLE
    (Causal Attention with Lookahead Keys) attention mechanism.
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
        
        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 6, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.combine_heads = LinearNoBias(dim_inner, dim)
    
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape
        
        # Project to all qkv components
        qkvs = self.to_all_qkv(x)
        
        # Split heads using einops
        qu, ku, vu, qc, kc, vc = self.split_heads(qkvs)
        
        # Apply Castle attention
        out = castle_attention(qc, kc, vc, qu, ku, vu, self.scale)
        
        # Merge heads using einops
        out = self.merge_heads(out)
        
        # Final projection
        out = self.combine_heads(out)
        
        return out