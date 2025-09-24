<img src="./fig3.png" width="400px"></img>

## Lookahead Keys Attention (wip)

Causal Attention with [Lookahead Keys](https://arxiv.org/abs/2509.07301)

## Installation

```bash
pip install lookahead-keys-attention
```

## Usage

```python
import torch
from lookahead_keys_attention import Castle

# Initialize the Castle attention module
model = Castle(
    dim=512,           # input dimension
    heads=8,           # number of attention heads
    dim_head=64,       # dimension per head
    use_triton=None    # auto-detect CUDA for Triton optimization
)

# Example with CUDA sequence
batch_size = 2
seq_len = 128
dim = 512

# Move to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Input sequence
x = torch.randn(batch_size, seq_len, dim).to(device)

# Forward pass
output = model(x)  # Shape: [batch_size, seq_len, dim]

# For inference with caching (single token generation)
cache = None
for i in range(seq_len):
    token = x[:, i:i+1, :]  # Single token
    output, cache = model(token, cache=cache, return_next_cache=True)
```

## Char level Enwik8

Make sure `uv` is installed (`pip install uv`)

Then

```shell
$ uv run train_triton.py
```

## Citations

```bibtex
@inproceedings{Song2025CausalAW,
    title   = {Causal Attention with Lookahead Keys},
    author  = {Zhuoqing Song and Peng Sun and Huizhuo Yuan and Quanquan Gu},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:281218151}
}
```
