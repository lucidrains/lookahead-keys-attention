import torch
import pytest

from lookahead_keys_attention.lookahead_keys_attention import (
    LookaheadKeysAttention,
    ParallelSlowCastle
)

def test_lookahead_keys_attention_runs():
    model = LookaheadKeysAttention()
    x = torch.randn(2, 10, 64)
    output = model(x)

def test_castle():
    batch_size = 1
    seq_len = 8
    dim = 32
    dim_head = 16
    heads = 2

    model = ParallelSlowCastle(dim=dim, dim_head=dim_head, heads=heads)
    model.eval()

    input_sequence = torch.randn(batch_size, seq_len, dim)

    cache = None
    recurrent_outputs = []
    with torch.no_grad():
        for t in range(seq_len):
            x_t = input_sequence[:, t:t+1, :]
            
            output_t, cache = model.forward(x_t, cache=cache)
            recurrent_outputs.append(output_t)

    final_recurrent_output = torch.cat(recurrent_outputs, dim=1)

    with torch.no_grad():
        output_parallel, _ = model.forward(input_sequence)

    assert final_recurrent_output.shape == output_parallel.shape
    assert torch.allclose(final_recurrent_output, output_parallel, atol=1e-6)
