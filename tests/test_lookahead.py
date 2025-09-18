import torch
import pytest

from lookahead_keys_attention.lookahead_keys_attention import (
    ParallelSlowCastle
)
from lookahead_keys_attention.lookahead_keys_attention_pytorch import (
    TritonCastleAttention
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

    parallel_part_output, cache = model(input_sequence[:, :split, :], return_next_cache = True)

    # naive sequential

    recurrent_outputs = []

    for t in range(split, seq_len):
        x_t = input_sequence[:, t:t+1, :]
        
        output_t, cache = model(x_t, cache = cache, return_next_cache = True)
        recurrent_outputs.append(output_t)

    recurrent_outputs = torch.cat(recurrent_outputs, dim = 1)

    final_recurrent_output = torch.cat((parallel_part_output, recurrent_outputs), dim = 1)

    # naive parallel

    output_parallel = model(input_sequence)

    assert final_recurrent_output.shape == output_parallel.shape

    assert torch.allclose(final_recurrent_output, output_parallel, atol = 1e-6)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_equivalence():
    batch_size = 2
    seq_len = 128
    dim = 32
    dim_head = 16
    heads = 2

    # define models
    
    naive_model = ParallelSlowCastle(dim=dim, dim_head=dim_head, heads=heads).cuda()
    triton_model = TritonCastleAttention(dim=dim, dim_head=dim_head, heads=heads).cuda()

    # copy weights from naive to triton model
    triton_model.to_all_qkv.weight.data.copy_(naive_model.to_all_qkv.weight.data)
    triton_model.combine_heads.weight.data.copy_(naive_model.combine_heads.weight.data)

    # inputs
    
    inp = torch.randn(batch_size, seq_len, dim).cuda()
    inp.requires_grad_()

    # forward pass
    
    naive_output = naive_model(inp)
    triton_output = triton_model(inp)

    assert torch.allclose(naive_output, triton_output, atol = 1e-3), "Forward outputs do not match"

    # backward pass
    
    grad_output = torch.randn_like(naive_output)
    
    naive_output.backward(grad_output, retain_graph=True)
    naive_grads = {name: p.grad.clone() for name, p in naive_model.named_parameters() if p.grad is not None}
    naive_input_grad = inp.grad.clone()

    inp.grad.zero_()
    for p in triton_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    triton_output.backward(grad_output, retain_graph=True)
    triton_grads = {name: p.grad.clone() for name, p in triton_model.named_parameters() if p.grad is not None}
    triton_input_grad = inp.grad.clone()

    # compare gradients

    assert torch.allclose(naive_input_grad, triton_input_grad, atol = 1e-3), "Input gradients do not match"

    for name in naive_grads.keys():
        assert name in triton_grads, f"Gradient for {name} not found in Triton model"
        assert torch.allclose(naive_grads[name], triton_grads[name], atol = 1e-3), f"Gradients for {name} do not match"