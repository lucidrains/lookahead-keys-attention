import torch
from torch import nn
from torch.nn import Module
from torch.autograd import Function

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