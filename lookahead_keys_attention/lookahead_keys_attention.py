from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, einsum
from torch.nn import Module
from torch.autograd import Function
import torch.nn.functional as F

from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(nn.Linear, bias=False)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# attention function

class LookaheadKeysAttentionFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# main module

class LookaheadKeysAttention(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LookaheadKeysAttentionFunction.apply(x)

# naive castle implementation

class ParallelSlowCastle(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads = 8
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads

        self.scale = dim_head ** -0.5

        self.w_q_u = LinearNoBias(dim, dim_inner)
        self.w_k_u = LinearNoBias(dim, dim_inner)
        self.w_v_u = LinearNoBias(dim, dim_inner)
        self.w_q_c = LinearNoBias(dim, dim_inner)
        self.w_k_c = LinearNoBias(dim, dim_inner)
        self.w_v_c = LinearNoBias(dim, dim_inner)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.combine_heads = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        x
    ):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        Q_u, K_u, V_u = self.w_q_u(x), self.w_k_u(x), self.w_v_u(x)
        Q_c, K_c, V_c = self.w_q_c(x), self.w_k_c(x), self.w_v_c(x)

        (Q_u, K_u, V_u, Q_c, K_c, V_c) = map(self.split_heads, (Q_u, K_u, V_u, Q_c, K_c, V_c))

        causal_mask = torch.triu(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=1)
        lookahead_mask = torch.tril(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=0)
        binary_causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

        term1 = (einsum('...id,...jd->...ij', Q_c, V_u) * self.scale) * binary_causal_mask
        sigmoid_term = torch.sigmoid((einsum('...id,...jd->...ij', Q_u, K_u) * self.scale) + lookahead_mask)
        S_u = einsum('...ij,...kj->...ik', term1, sigmoid_term)

        S_c = (einsum('...id,...jd->...ij', Q_c, K_c) * self.scale) + causal_mask
        
        final_scores = S_c - F.silu(S_u)
        
        attention_weights = F.softmax(final_scores, dim=-1)
        output = einsum('...ij,...jd->...id', attention_weights, V_c)

        output = self.merge_heads(output)

        return self.combine_heads(output)

    def forward_inference(
        self,
        x_t,
        cache: dict | None = None
    ):
        batch_size, _, _ = x_t.shape
        device = x_t.device

        if not exists(cache):
            shape = (batch_size, self.heads, 0, self.dim_head)

            cache = {
                'U': torch.empty(shape, device=device),
                'Q_u': torch.empty(shape, device=device),
                'K_c': torch.empty(shape, device=device),
                'V_c': torch.empty(shape, device=device),
            }

        U_prev, Q_u_cache, K_c_prev, V_c_prev = cache['U'], cache['Q_u'], cache['K_c'], cache['V_c']

        q_t_u, k_t_u, v_t_u = self.w_q_u(x_t), self.w_k_u(x_t), self.w_v_u(x_t)
        q_t_c, k_t_c, v_t_c = self.w_q_c(x_t), self.w_k_c(x_t), self.w_v_c(x_t)

        (q_t_u, k_t_u, v_t_u, q_t_c, k_t_c, v_t_c) = map(self.split_heads, (q_t_u, k_t_u, v_t_u, q_t_c, k_t_c, v_t_c))

        update_weights = torch.sigmoid(torch.einsum('...id,...jd->...ij', Q_u_cache, k_t_u) * self.scale)

        update_values = update_weights * v_t_u
        U_updated_prev = U_prev + update_values

        u_t_lookahead = torch.zeros_like(q_t_u)
        U_t = cat([U_updated_prev, u_t_lookahead], dim=-2)

        K_c_t = cat([K_c_prev, k_t_c], dim=-2)
        V_c_t = cat([V_c_prev, v_t_c], dim=-2)

        s_t_c = einsum('...id,...jd->...ij', q_t_c, K_c_t) * self.scale
        s_t_u = einsum('...id,...jd->...ij', q_t_c, U_t) * self.scale
        p_t = F.softmax(s_t_c - F.silu(s_t_u), dim=-1)
        output_t = einsum('...ij,...jd->...id', p_t, V_c_t)
        
        output_t = self.merge_heads(output_t)

        next_cache = {
            'U': U_t,
            'Q_u': cat([Q_u_cache, q_t_u], dim=-2),
            'K_c': K_c_t,
            'V_c': V_c_t,
        }
        
        return self.combine_heads(output_t), next_cache
