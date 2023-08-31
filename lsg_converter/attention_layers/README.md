# Block-Local-Self-Attention
This is a substitute of vanilla Self Attention (bidirectionnal or causal). \
Doesn't work for Cross Attention because the local context is ambiguous to define in this case. 

## Usage
In causal modeling, each query is connected up to `2*block_size` keys (+ global). \
In non causal modeling, each query is connected up to `3*block_size` keys (+ global).

```python
from lsg_converter.attention_layers import BlockLocalSelfAttention

# batch, num_heads, sequence_length, hidden_size
n, h, t, d = 2, 4, 58, 32  

Q, K, V = torch.randn(n, h, t, d), torch.randn(n, h, t, d), torch.randn(n, h, t, d)

# attention_mask is 0 for no mask, -inf for mask (similar to most HuggingFace models)
attention_mask = torch.zeros(n, 1, 1, t).float()

attn = BlockLocalSelfAttention(block_size=16, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1)

# expect (batch, num_heads, sequence_length, hidden_size) inputs,
# attention_mask is (batch, 1, 1, sequence_length) 
# causal mask is built on the fly but (batch, 1, sequence_length, sequence_length) mask is possible
outputs = attn(Q, K, V, attention_mask)

print(outputs.shape)
> torch.Size([2, 4, 58, 32])
```

Replacing Self Attention in GPT2 (from Huggingface):
```python
from transformers.models.gpt2 import * 
from lsg_converter.attention_layers import BlockLocalSelfAttention

class GPT2BlockLocalAttention(modeling_gpt2.GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.attn = BlockLocalSelfAttention(block_size=32, compute_global_attention=True, is_causal=True)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        return self.attn(query, key, value, attention_mask), None
    
modeling_gpt2.GPT2Attention = GPT2BlockLocalAttention
```
Note that for generation (inference on causal modeling), full attention is used after the first step. \
This may change in the future.

## Parameters
* `config` (default None): a configuration, not used by default
* `block_size` (default 128): size of the local block
* `compute_global_attention` (default True): add a global connection to the first token
* `is_causal` (default False): for causal modeling
* `attention_dropout_prob` (default 0.1): attention dropout
* `preprocessing_function` (default None): replaces the `.preprocess(...)` method (i.e reshaping Q, K, V, mask etc...) \
    * `arguments`: all the .forward(...) arguments `(query_layer, key_layer, value_layer, attention_mask, **kwargs)` \
    * `returns`: must return `(query_layer, key_layer, value_layer, attention_mask)`

## Methods for customization
* `.post_init()` which is run at the end of `__init__(...)`
* `.preprocess(query_layer, key_layer, value_layer, attention_mask, **kwargs)` that behaves like `preprocessing_function`

## Inputs and outputs
The `.forward(...)` methods at least 4 arguments which are passed to the `.preprocess(...)` methods:
* `query_layer`: usually a `torch.tensor` of shape `(batch, num_heads, sequence_length, hidden_size)`
* `key_layer`: usually a `torch.tensor` of shape `(batch, num_heads, sequence_length, hidden_size)`
* `value_layer`: usually a `torch.tensor` of shape `(batch, num_heads, sequence_length, hidden_size)`
* `attention_mask`: usually a `torch.tensor` of shape `(batch, 1, 1, sequence_length)` where `0` is non mask and `-inf` is mask
* `**kwargs**`: additional arguments

The output is a `torch.tensor` with the same shape as `query_layer`
