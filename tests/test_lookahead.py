import torch
import pytest
from lookahead_keys_attention.lookahead_keys_attention import LookaheadKeysAttention


def test_lookahead_keys_attention_runs():
    model = LookaheadKeysAttention()
    x = torch.randn(2, 10, 64)
    output = model(x)