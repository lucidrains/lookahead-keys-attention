from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, einsum
from torch.nn import Module
from torch.autograd import Function
import torch.nn.functional as F

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
        dim_head
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.w_q_u = LinearNoBias(dim, dim_head)
        self.w_k_u = LinearNoBias(dim, dim_head)
        self.w_v_u = LinearNoBias(dim, dim_head)
        self.w_q_c = LinearNoBias(dim, dim_head)
        self.w_k_c = LinearNoBias(dim, dim_head)
        self.w_v_c = LinearNoBias(dim, dim_head)

    def forward(
        self,
        x
    ):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        Q_u, K_u, V_u = self.w_q_u(x), self.w_k_u(x), self.w_v_u(x)
        Q_c, K_c, V_c = self.w_q_c(x), self.w_k_c(x), self.w_v_c(x)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=1)
        lookahead_mask = torch.tril(torch.full((seq_len, seq_len), -torch.inf, device=device), diagonal=0)
        binary_causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

        term1 = (einsum('bid,bjd->bij', Q_c, V_u) * self.scale) * binary_causal_mask
        sigmoid_term = torch.sigmoid((einsum('bid,bjd->bij', Q_u, K_u) * self.scale) + lookahead_mask)
        S_u = einsum('bij,bkj->bik', term1, sigmoid_term)

        S_c = (einsum('bid,bjd->bij', Q_c, K_c) * self.scale) + causal_mask
        
        final_scores = S_c - F.silu(S_u)
        
        attention_weights = F.softmax(final_scores, dim=-1)
        output = einsum('bij,bjd->bid', attention_weights, V_c)

        return output

    def forward_inference(
        self,
        x_t,
        cache: dict | None = None
    ):
        batch_size, _, _ = x_t.shape

        if not exists(cache):
            cache = {
                'U': torch.empty(batch_size, 0, self.dim_head, device=x_t.device),
                'Q_u': torch.empty(batch_size, 0, self.dim_head, device=x_t.device),
                'K_c': torch.empty(batch_size, 0, self.dim_head, device=x_t.device),
                'V_c': torch.empty(batch_size, 0, self.dim_head, device=x_t.device),
            }

        U_prev, Q_u_cache, K_c_prev, V_c_prev = cache['U'], cache['Q_u'], cache['K_c'], cache['V_c']

        q_t_u, k_t_u, v_t_u = self.w_q_u(x_t), self.w_k_u(x_t), self.w_v_u(x_t)
        q_t_c, k_t_c, v_t_c = self.w_q_c(x_t), self.w_k_c(x_t), self.w_v_c(x_t)

        update_weights = torch.sigmoid(torch.einsum('bid,bjd->bij', Q_u_cache, k_t_u) * self.scale)

        update_values = update_weights * v_t_u
        U_updated_prev = U_prev + update_values

        u_t_lookahead = torch.zeros_like(q_t_u)
        U_t = cat([U_updated_prev, u_t_lookahead], dim=1)

        K_c_t = cat([K_c_prev, k_t_c], dim=1)
        V_c_t = cat([V_c_prev, v_t_c], dim=1)

        s_t_c = einsum('bid,bjd->bij', q_t_c, K_c_t) * self.scale
        s_t_u = einsum('bid,bjd->bij', q_t_c, U_t) * self.scale
        p_t = F.softmax(s_t_c - F.silu(s_t_u), dim=-1)
        output_t = einsum('bij,bjd->bid', p_t, V_c_t)
        
        next_cache = {
            'U': U_t,
            'Q_u': cat([Q_u_cache, q_t_u], dim=1),
            'K_c': K_c_t,
            'V_c': V_c_t,
        }
        
        return output_t, next_cache
