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

class LookaheadKeysAttention(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# naive castle implementation with improved DRY forward method

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
        x,
        cache: dict | None = None
    ):
        batch_size, seq_len, device = *x.shape[:2], x.device

        q_u, k_u, v_u = self.w_q_u(x), self.w_k_u(x), self.w_v_u(x)
        q_c, k_c, v_c = self.w_q_c(x), self.w_k_c(x), self.w_v_c(x)

        (q_u, k_u, v_u, q_c, k_c, v_c) = map(self.split_heads, (q_u, k_u, v_u, q_c, k_c, v_c))

        is_inference = seq_len == 1

        if not is_inference:

            causal_mask = torch.triu(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=1)
            lookahead_mask = torch.tril(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=0)
            binary_causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

            term1 = (einsum('...id,...jd->...ij', q_c, v_u) * self.scale) * binary_causal_mask
            sigmoid_term = torch.sigmoid((einsum('...id,...jd->...ij', q_u, k_u) * self.scale) + lookahead_mask)
            S_u = einsum('...ij,...kj->...ik', term1, sigmoid_term)
            S_c = (einsum('...id,...jd->...ij', q_c, k_c) * self.scale) + causal_mask
            
            final_scores = S_c - F.silu(S_u)
            
            V_c_context = v_c
            next_cache = None

        else:

            if not exists(cache):
                shape = (batch_size, self.heads, 0, self.dim_head)

                cache = dict(
                    U = torch.empty(shape, device=device),
                    Q_u = torch.empty(shape, device=device),
                    K_c = torch.empty(shape, device=device),
                    V_c = torch.empty(shape, device=device)
                )

            U_prev, Q_u_cache, K_c_prev, V_c_prev = cache['U'], cache['Q_u'], cache['K_c'], cache['V_c']

            update_weights = torch.sigmoid(torch.einsum('...id,...jd->...ij', Q_u_cache, k_u) * self.scale)
            U_updated_prev = U_prev + (update_weights * v_u)
            U_t = cat([U_updated_prev, torch.zeros_like(q_u)], dim=-2)

            K_c_context = cat([K_c_prev, k_c], dim=-2)
            V_c_context = cat([V_c_prev, v_c], dim=-2)

            s_t_c = einsum('...id,...jd->...ij', q_c, K_c_context) * self.scale
            s_t_u = einsum('...id,...jd->...ij', q_c, U_t) * self.scale
            final_scores = s_t_c - F.silu(s_t_u)

            next_cache = dict(
                U = U_t,
                Q_u = cat([Q_u_cache, q_u], dim=-2),
                K_c = K_c_context,
                V_c = V_c_context
            )

        attention_weights = F.softmax(final_scores, dim=-1)
        output = einsum('...ij,...jd->...id', attention_weights, V_c_context)

        output = self.merge_heads(output)
        return self.combine_heads(output), next_cache
