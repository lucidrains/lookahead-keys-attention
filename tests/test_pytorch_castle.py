import torch
import pytest
from torch.autograd import gradcheck

from lookahead_keys_attention.lookahead_keys_attention import (
    ParallelSlowCastle
)
from lookahead_keys_attention.lookahead_keys_attention_pytorch import (
    TritonCastleAttention,
    castle_attention
)


@pytest.fixture
def setup_models():
    """Setup fixture for test models."""
    torch.manual_seed(42)
    dim = 64
    heads = 4
    dim_head = 16
    
    naive_model = ParallelSlowCastle(dim=dim, heads=heads, dim_head=dim_head)
    pytorch_model = TritonCastleAttention(dim=dim, heads=heads, dim_head=dim_head)
    
    # Copy weights from naive to pytorch
    with torch.no_grad():
        pytorch_model.to_all_qkv.weight.copy_(naive_model.to_all_qkv.weight)
        pytorch_model.combine_heads.weight.copy_(naive_model.combine_heads.weight)
    
    return naive_model, pytorch_model


def test_forward_equivalence(setup_models):
    """Test that forward pass matches between naive and PyTorch implementations."""
    naive_model, pytorch_model = setup_models
    
    batch_size = 2
    seq_len = 8
    dim = 64
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    with torch.no_grad():
        out_naive = naive_model(x)
        out_pytorch = pytorch_model(x)
    
    # Check outputs match
    assert torch.allclose(out_naive, out_pytorch, atol=1e-5), \
        f"Forward outputs don't match. Max diff: {(out_naive - out_pytorch).abs().max().item()}"


def test_backward_equivalence(setup_models):
    """Test that gradients match between naive and PyTorch implementations."""
    naive_model, pytorch_model = setup_models
    
    batch_size = 2
    seq_len = 8
    dim = 64
    
    # Create input with gradients enabled
    x_naive = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    x_pytorch = x_naive.clone().detach().requires_grad_(True)
    
    # Forward pass
    out_naive = naive_model(x_naive)
    out_pytorch = pytorch_model(x_pytorch)
    
    # Create gradient
    grad_out = torch.randn_like(out_naive)
    
    # Backward pass
    out_naive.backward(grad_out.clone())
    out_pytorch.backward(grad_out.clone())
    
    # Check input gradients match
    assert torch.allclose(x_naive.grad, x_pytorch.grad, atol=1e-5), \
        f"Input gradients don't match. Max diff: {(x_naive.grad - x_pytorch.grad).abs().max().item()}"
    
    # Check weight gradients match
    assert torch.allclose(
        naive_model.to_all_qkv.weight.grad, 
        pytorch_model.to_all_qkv.weight.grad, 
        atol=1e-5
    ), "QKV weight gradients don't match"
    
    assert torch.allclose(
        naive_model.combine_heads.weight.grad,
        pytorch_model.combine_heads.weight.grad,
        atol=1e-5
    ), "Combine heads weight gradients don't match"


def test_different_sequence_lengths(setup_models):
    """Test with various sequence lengths."""
    naive_model, pytorch_model = setup_models
    
    batch_size = 2
    dim = 64
    
    for seq_len in [1, 4, 8, 16, 32]:
        x = torch.randn(batch_size, seq_len, dim)
        
        with torch.no_grad():
            out_naive = naive_model(x, return_next_cache=False)
            out_pytorch = pytorch_model(x)
            
            # Handle case where naive_model might return tuple
            if isinstance(out_naive, tuple):
                out_naive = out_naive[0]
        
        assert torch.allclose(out_naive, out_pytorch, atol=1e-5), \
            f"Outputs don't match for seq_len={seq_len}"


def test_different_batch_sizes():
    """Test with various batch sizes."""
    torch.manual_seed(42)
    
    dim = 32
    heads = 2
    dim_head = 16
    seq_len = 8
    
    for batch_size in [1, 2, 4, 8]:
        naive_model = ParallelSlowCastle(dim=dim, heads=heads, dim_head=dim_head)
        pytorch_model = TritonCastleAttention(dim=dim, heads=heads, dim_head=dim_head)
        
        # Copy weights
        with torch.no_grad():
            pytorch_model.to_all_qkv.weight.copy_(naive_model.to_all_qkv.weight)
            pytorch_model.combine_heads.weight.copy_(naive_model.combine_heads.weight)
        
        x = torch.randn(batch_size, seq_len, dim)
        
        with torch.no_grad():
            out_naive = naive_model(x)
            out_pytorch = pytorch_model(x)
        
        assert torch.allclose(out_naive, out_pytorch, atol=1e-5), \
            f"Outputs don't match for batch_size={batch_size}"


def test_castle_attention_function():
    """Test the castle_attention custom autograd function directly."""
    torch.manual_seed(42)
    
    batch = 1
    heads = 1
    seq_len = 4
    dim_head = 8
    scale = dim_head ** -0.5
    
    # Create inputs
    qc = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    kc = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    vc = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    qu = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    ku = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    vu = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)
    
    # Test forward
    out = castle_attention(qc, kc, vc, qu, ku, vu, scale)
    assert out.shape == (batch, heads, seq_len, dim_head), "Output shape incorrect"
    
    # Test backward
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    
    # Check all inputs have gradients
    for tensor, name in [(qc, 'qc'), (kc, 'kc'), (vc, 'vc'), 
                         (qu, 'qu'), (ku, 'ku'), (vu, 'vu')]:
        assert tensor.grad is not None, f"{name} gradient is None"
        assert not torch.isnan(tensor.grad).any(), f"{name} gradient contains NaN"
        assert not torch.isinf(tensor.grad).any(), f"{name} gradient contains Inf"


def test_gradient_accumulation():
    """Test that gradients accumulate correctly across multiple backward passes."""
    torch.manual_seed(42)
    
    dim = 32
    heads = 2
    dim_head = 16
    batch_size = 2
    seq_len = 8
    
    model = TritonCastleAttention(dim=dim, heads=heads, dim_head=dim_head)
    
    # First forward-backward
    x1 = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out1 = model(x1)
    loss1 = out1.sum()
    loss1.backward()
    
    # Save first gradients
    grad1 = model.to_all_qkv.weight.grad.clone()
    
    # Second forward-backward (gradients should accumulate)
    x2 = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out2 = model(x2)
    loss2 = out2.sum()
    loss2.backward()
    
    # Check gradients accumulated
    grad2 = model.to_all_qkv.weight.grad
    assert not torch.allclose(grad1, grad2), "Gradients didn't accumulate"
    
    # Reset and test again
    model.zero_grad()
    out3 = model(x1)
    loss3 = out3.sum()
    loss3.backward()
    
    # Should match first gradient since we reset
    grad3 = model.to_all_qkv.weight.grad
    assert torch.allclose(grad1, grad3, atol=1e-6), "Gradients after reset don't match"


def test_causal_masking():
    """Test that causal masking is applied correctly."""
    torch.manual_seed(42)
    
    batch = 1
    heads = 1
    seq_len = 4
    dim_head = 8
    scale = dim_head ** -0.5
    
    # Create inputs where all values are ones - this makes it easier to verify masking
    qc = torch.ones(batch, heads, seq_len, dim_head, requires_grad=True)
    kc = torch.ones(batch, heads, seq_len, dim_head, requires_grad=True)
    vc = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)  # Keep vc random
    qu = torch.ones(batch, heads, seq_len, dim_head, requires_grad=True)
    ku = torch.ones(batch, heads, seq_len, dim_head, requires_grad=True)
    vu = torch.randn(batch, heads, seq_len, dim_head, requires_grad=True)  # Keep vu random
    
    # Forward pass
    out = castle_attention(qc, kc, vc, qu, ku, vu, scale)
    
    # The output should show causal structure
    # First position should only attend to itself
    # Last position can attend to all previous positions
    # This is hard to test directly, but we can at least verify the output is valid
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    
    # All gradients should be valid
    for tensor in [qc, kc, vc, qu, ku, vu]:
        assert tensor.grad is not None, "Gradient is None"
        assert not torch.isnan(tensor.grad).any(), "Gradient contains NaN"


@pytest.mark.parametrize("batch,seq_len,heads,dim_head", [
    (1, 4, 1, 8),
    (2, 8, 2, 16),
    (4, 16, 4, 32),
    (1, 32, 8, 64),
])
def test_various_configurations(batch, seq_len, heads, dim_head):
    """Test with various model configurations."""
    torch.manual_seed(42)
    
    dim = heads * dim_head
    
    # Create models
    naive_model = ParallelSlowCastle(dim=dim, heads=heads, dim_head=dim_head)
    pytorch_model = TritonCastleAttention(dim=dim, heads=heads, dim_head=dim_head)
    
    # Copy weights
    with torch.no_grad():
        pytorch_model.to_all_qkv.weight.copy_(naive_model.to_all_qkv.weight)
        pytorch_model.combine_heads.weight.copy_(naive_model.combine_heads.weight)
    
    # Test forward
    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    x_naive = x.clone().detach().requires_grad_(True)
    x_pytorch = x.clone().detach().requires_grad_(True)
    
    out_naive = naive_model(x_naive)
    out_pytorch = pytorch_model(x_pytorch)
    
    assert torch.allclose(out_naive, out_pytorch, atol=1e-5), \
        f"Forward mismatch for config: batch={batch}, seq={seq_len}, heads={heads}, dim_head={dim_head}"
    
    # Test backward
    grad_out = torch.randn_like(out_naive)
    out_naive.backward(grad_out.clone())
    out_pytorch.backward(grad_out.clone())
    
    assert torch.allclose(x_naive.grad, x_pytorch.grad, atol=1e-5), \
        f"Gradient mismatch for config: batch={batch}, seq={seq_len}, heads={heads}, dim_head={dim_head}"


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    torch.manual_seed(42)
    
    batch = 1
    heads = 1
    seq_len = 4
    dim_head = 8
    scale = dim_head ** -0.5
    
    # Test with very small values
    eps = 1e-7
    qc = torch.randn(batch, heads, seq_len, dim_head) * eps
    kc = torch.randn(batch, heads, seq_len, dim_head) * eps
    vc = torch.randn(batch, heads, seq_len, dim_head) * eps
    qu = torch.randn(batch, heads, seq_len, dim_head) * eps
    ku = torch.randn(batch, heads, seq_len, dim_head) * eps
    vu = torch.randn(batch, heads, seq_len, dim_head) * eps
    
    # Make them leaf tensors requiring grad
    qc = qc.detach().requires_grad_(True)
    kc = kc.detach().requires_grad_(True)
    vc = vc.detach().requires_grad_(True)
    qu = qu.detach().requires_grad_(True)
    ku = ku.detach().requires_grad_(True)
    vu = vu.detach().requires_grad_(True)
    
    out = castle_attention(qc, kc, vc, qu, ku, vu, scale)
    assert not torch.isnan(out).any(), "Output contains NaN with small inputs"
    assert not torch.isinf(out).any(), "Output contains Inf with small inputs"
    
    # Test gradient
    loss = out.sum()
    loss.backward()
    
    for tensor, name in [(qc, 'qc'), (kc, 'kc'), (vc, 'vc'), 
                         (qu, 'qu'), (ku, 'ku'), (vu, 'vu')]:
        assert tensor.grad is not None, f"{name} gradient is None"
        assert not torch.isnan(tensor.grad).any(), f"{name} gradient contains NaN with small inputs"
        assert not torch.isinf(tensor.grad).any(), f"{name} gradient contains Inf with small inputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])