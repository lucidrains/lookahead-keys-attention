import torch
import pytest

from lookahead_keys_attention.lookahead_keys_attention import (
    ParallelSlowCastle
)

@torch.no_grad()
def test_naive_castle():
    batch_size = 2
    seq_len = 16
    dim = 32
    dim_head = 16
    heads = 2
    split = 8

    # define

    model = ParallelSlowCastle(dim=dim, dim_head=dim_head, heads=heads)
    model.eval()

    input_sequence = torch.randn(batch_size, seq_len, dim)

    # initial parallel

    parallel_part_output, cache = model.forward(input_sequence[:, :split, :], return_next_cache = True)

    # naive sequential

    recurrent_outputs = []

    for t in range(split, seq_len):
        x_t = input_sequence[:, t:t+1, :]
        
        output_t, cache = model.forward(x_t, cache = cache, return_next_cache = True)
        recurrent_outputs.append(output_t)

    recurrent_outputs = torch.cat(recurrent_outputs, dim = 1)

    final_recurrent_output = torch.cat((parallel_part_output, recurrent_outputs), dim = 1)

    # naive parallel

    with torch.no_grad():
        output_parallel = model.forward(input_sequence)

    assert final_recurrent_output.shape == output_parallel.shape
    assert torch.allclose(final_recurrent_output, output_parallel, atol=1e-6)
