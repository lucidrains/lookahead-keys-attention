import torch
import pytest

from lookahead_keys_attention.lookahead_keys_attention import (
    Castle
)

@torch.no_grad()
def test_castle_reference_implementation():
    """Test Castle with reference PyTorch implementation (use_triton=False)"""
    batch_size = 2
    seq_len = 16
    dim = 32
    dim_head = 16
    heads = 2
    split = 8

    # define - explicitly use reference implementation
    model = Castle(dim=dim, dim_head=dim_head, heads=heads, use_triton=False)
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
def test_castle_triton_vs_reference():
    """Test Castle with Triton implementation vs reference implementation"""
    batch_size = 2
    seq_len = 128
    dim = 32
    dim_head = 16
    heads = 2

    # define models
    reference_model = Castle(dim=dim, dim_head=dim_head, heads=heads, use_triton=False).cuda()
    triton_model = Castle(dim=dim, dim_head=dim_head, heads=heads, use_triton=True).cuda()

    # copy weights from reference to triton model
    triton_model.to_all_qkv.weight.data.copy_(reference_model.to_all_qkv.weight.data)
    triton_model.combine_heads.weight.data.copy_(reference_model.combine_heads.weight.data)

    # inputs
    
    inp = torch.randn(batch_size, seq_len, dim).cuda()
    inp.requires_grad_()

    # forward pass

    reference_output = reference_model(inp)
    triton_output = triton_model(inp)

    assert torch.allclose(reference_output, triton_output, atol = 1e-3), "Forward outputs do not match"

    # backward pass

    grad_output = torch.randn_like(reference_output)

    reference_output.backward(grad_output, retain_graph=True)
    reference_grads = {name: p.grad.clone() for name, p in reference_model.named_parameters() if p.grad is not None}
    reference_input_grad = inp.grad.clone()

    inp.grad.zero_()
    for p in triton_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    triton_output.backward(grad_output, retain_graph=True)
    triton_grads = {name: p.grad.clone() for name, p in triton_model.named_parameters() if p.grad is not None}
    triton_input_grad = inp.grad.clone()

    # compare gradients

    assert torch.allclose(reference_input_grad, triton_input_grad, atol = 1e-2), "Input gradients do not match"

    for name in reference_grads.keys():
        assert name in triton_grads, f"Gradient for {name} not found in Triton model"
        assert torch.allclose(reference_grads[name], triton_grads[name], atol = 1e-2), f"Gradients for {name} do not match"


