import torch
import pytest

from lookahead_keys_attention.lookahead_keys_attention import (
    Castle
)

param = pytest.mark.parametrize

@torch.no_grad()
@param('prenorm', (False, True))
def test_castle_reference_implementation(
    prenorm
):
    """Test Castle with reference PyTorch implementation (use_triton=False)"""
    batch_size = 2
    seq_len = 16
    dim = 32
    dim_head = 16
    heads = 2
    split = 8

    # define - explicitly use reference implementation

    model = Castle(
        dim = dim,
        dim_head = dim_head,
        heads = heads,
        prenorm = prenorm,
        use_triton = False
    )

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
def test_castle_triton_vs_reference():
    """Test Castle with Triton implementation vs reference implementation"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available")
    
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

@torch.no_grad()
def test_castle_with_rotary_embeddings():
    """Test Castle with rotary embeddings enabled"""
    batch_size = 2
    seq_len = 16
    dim = 256
    dim_head = 32
    heads = 8
    split = 8

    # Create models with and without rotary embeddings
    model_no_rotary = Castle(dim=dim, dim_head=dim_head, heads=heads, rotary_emb=False, use_triton=False)
    model_with_rotary = Castle(dim=dim, dim_head=dim_head, heads=heads, rotary_emb=True, use_triton=False)
    
    # Copy weights to ensure only rotary embeddings cause differences
    model_with_rotary.to_all_qkv.weight.data.copy_(model_no_rotary.to_all_qkv.weight.data)
    model_with_rotary.combine_heads.weight.data.copy_(model_no_rotary.combine_heads.weight.data)
    
    model_no_rotary.eval()
    model_with_rotary.eval()

    input_sequence = torch.randn(batch_size, seq_len, dim)

    # Test parallel execution
    output_no_rotary = model_no_rotary(input_sequence)
    output_with_rotary = model_with_rotary(input_sequence)
    
    # Outputs should have same shape
    assert output_no_rotary.shape == output_with_rotary.shape == (batch_size, seq_len, dim)
    
    # Outputs should be different due to rotary embeddings
    assert not torch.allclose(output_no_rotary, output_with_rotary, atol=1e-6), \
        "Outputs should differ when rotary embeddings are applied"

    # Test sequential execution with rotary embeddings
    parallel_part_output, cache = model_with_rotary(
        input_sequence[:, :split, :], 
        return_next_cache=True
    )
    
    recurrent_outputs = []
    for t in range(split, seq_len):
        x_t = input_sequence[:, t:t+1, :]
        output_t, cache = model_with_rotary(x_t, cache=cache, return_next_cache=True)
        recurrent_outputs.append(output_t)
    
    recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
    final_recurrent_output = torch.cat((parallel_part_output, recurrent_outputs), dim=1)
    
    # Sequential and parallel should match for model with rotary (with slightly higher tolerance due to rotary)
    assert torch.allclose(final_recurrent_output, output_with_rotary, atol=1e-3), \
        "Sequential and parallel outputs should match with rotary embeddings"

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
def test_castle_rotary_with_triton():
    
    batch_size = 2
    seq_len = 64
    dim = 256
    dim_head = 32
    heads = 8

    # Create models
    model_reference = Castle(
        dim=dim, dim_head=dim_head, heads=heads, 
        rotary_emb=True, use_triton=False
    ).cuda()
    
    model_triton = Castle(
        dim=dim, dim_head=dim_head, heads=heads,
        rotary_emb=True, use_triton=True
    ).cuda()
    
    # Copy weights
    model_triton.to_all_qkv.weight.data.copy_(model_reference.to_all_qkv.weight.data)
    model_triton.combine_heads.weight.data.copy_(model_reference.combine_heads.weight.data)
    
    # Test forward pass
    inp = torch.randn(batch_size, seq_len, dim).cuda()
    
    output_reference = model_reference(inp)
    output_triton = model_triton(inp)
    
    # Outputs should match between reference and triton with rotary
    assert torch.allclose(output_reference, output_triton, atol=1e-3), \
        "Reference and Triton outputs should match with rotary embeddings"
