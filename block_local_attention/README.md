# Block-Local-Attention

Usage:

```python
from block_local_attention import *

# batch, num_heads, sequence length, hidden_size
n, h, t, d = 2, 4, 58, 32  

Q, K, V = torch.randn(n, h, t, d), torch.randn(n, h, t, d), torch.randn(n, h, t, d)
attention_mask = torch.zeros(n, 1, 1, t).float()

attn = BlockLocalSelfAttention(block_size=16, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1)

# expect (n, h, t, d) inputs,
# attention_mask is (n, 1, 1, t) or (n, 1, t, t) for causal
# attention_mask is 0 for no mask, -inf for mask (similar to most HuggingFace models)
outputs = attn(Q, K, V, attention_mask)

print(outputs.shape)
> torch.Size([2, 4, 58, 32])
```
