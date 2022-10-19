import torch
import torch.nn as nn
import math

class BlockLocalSelfAttention(nn.Module):

    def __init__(self, block_size=128, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()

        self.compute_global_attention = compute_global_attention
        self.block_size = block_size
        self.is_causal = is_causal
        self.dropout = nn.Dropout(attention_dropout_prob)

        # Shape of blocks
        self.local_shapes = (self.block_size*3, self.block_size)

        
        if is_causal:
            self.attention = self.causal_attention_product
        else:
            self.attention = self.attention_product
        
    def forward(
        self, 
        query_layer, 
        key_layer, 
        value_layer, 
        attention_mask=None, 
        ):

        assert query_layer.size() == key_layer.size() == value_layer.size(), "Q, K, V have to be of the same size"

        n, h, t, d = query_layer.size()
        if attention_mask is None:
            attention_mask = torch.zeros(n, 1, 1, t, device=query_layer.device, dtype=query_layer.dtype)

        if t <= 2*self.block_size:
            return self.attention_product(query_layer, key_layer, value_layer, attention_mask)

        outputs = self.block_local_forward(query_layer, key_layer, value_layer, attention_mask)

        if self.compute_global_attention and not self.is_causal:
            outputs[..., :1, :] = self.attention_product(
                query_layer[..., :1, :], 
                key_layer, 
                value_layer, 
                attention_mask[..., :1, :]
                )

        return outputs 

    def block_local_forward(
        self, 
        query_layer, 
        key_layer, 
        value_layer, 
        attention_mask=None, 
        ):

        """
        Expects Q, K, V to be (n, h, t, d) == (batch, num_heads, sequence_length, hidden_size)
        attention_mask: (n, 1, 1, t) == (batch, 1, 1, sequence_length)  (-inf for mask, 0 else)
        """

        # Input batch, heads, length, hidden_size
        n, h, t, d = query_layer.size()

        # Pad if necessary
        extra_tokens = t % self.block_size
        pad = (0, self.block_size - extra_tokens)

        if extra_tokens > 0:
            query_layer = self.pad_inputs(query_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)
            key_layer = self.pad_inputs(key_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)
            value_layer = self.pad_inputs(value_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)

            pad = (self.block_size - extra_tokens, self.block_size - extra_tokens) if self.is_causal else pad
            attention_mask = self.pad_inputs(attention_mask, pad=pad, value=torch.finfo(attention_mask.dtype).min)

        n, h, t_, d = query_layer.size()
        n_blocks = t_ // self.block_size

        key_layer = self.build_lsg_inputs(
            key_layer, 
            key_layer[..., :1, :]
            )

        value_layer = self.build_lsg_inputs(
            value_layer, 
            value_layer[..., :1, :]
            )

        # Mask bos to avoid double connection
        global_mask = torch.zeros(n, 1, 1, attention_mask.size()[-2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask[..., 0] = torch.finfo(attention_mask.dtype).min 

        attention_mask = self.build_lsg_inputs(
            attention_mask, 
            global_mask,
            is_attn_mask=True
            ).transpose(-1, -2)

        # expect (..., t, d) shape
        # Compute attention
        context_layer = self.attention(
                query_layer=self.chunk(query_layer, n_blocks), 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                )
                
        return context_layer.reshape(n, h, -1, d)[..., :t, :]
    
    def attention_product(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Apply the attention mask is (precomputed for all layers in AlbertModel forward() function)
        attention_scores = attention_scores + attention_mask
        del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer

    def causal_attention_product(self, query_layer, key_layer, value_layer, attention_mask=None, causal_shape=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Apply the attention mask
        attention_scores = attention_scores + attention_mask[..., :1, :]

        # Add causal mask
        causal_shape = (self.block_size, self.block_size) if causal_shape is None else causal_shape
        causal_mask = torch.tril(
            torch.ones(*causal_shape, device=attention_mask.device, dtype=attention_scores.dtype), 
            diagonal=-1
            ) 
        causal_mask = causal_mask.T * torch.finfo(attention_scores.dtype).min
        attention_scores[..., -causal_shape[0]:, -causal_shape[1] + 1:] = causal_mask[:, 1:]
        del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer

    def build_lsg_inputs(self, hidden_states, global_hidden_states, is_attn_mask=False):
        
        return self.cat_tokens(
            global_hidden_states, 
            self.reshape_to_local_block(hidden_states, is_attn_mask)
            )

    def pad_inputs(self, inputs, pad, value=0):
        return torch.nn.functional.pad(inputs, pad=pad, value=value)

    def reshape_to_local_block(self, hidden_states, is_attn_mask=False):
        
        size, step = self.local_shapes
        s = (size - step) // 2

        # Pad before block reshaping
        if is_attn_mask:
            pad_value = torch.finfo(hidden_states.dtype).min 
            hidden_states = hidden_states.transpose(-1, -2)
        else: 
            pad_value = 0

        hidden_states = torch.nn.functional.pad(
            hidden_states.transpose(-1, -2), 
            pad=(s, s),
            value=pad_value
            ).transpose(-1, -2)

        # Make blocks
        hidden_states = hidden_states.unfold(-2, size=size, step=step).transpose(-1, -2)

        # Skip third block if causal
        if self.is_causal:
            return hidden_states[..., :size*2//3, :]

        return hidden_states

    def cat_tokens(self, x_global, x_local, dim=-2):

        n, h, b, t, d = x_local.size()
        x_global = x_global.unsqueeze(-3).expand(-1, -1, b, -1, -1)
        return torch.cat([x_global, x_local], dim=dim)

    def chunk(self, x, n_blocks):

        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], n_blocks, -1, d)

