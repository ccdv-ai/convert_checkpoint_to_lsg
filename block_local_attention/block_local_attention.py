import torch
import torch.nn as nn
import math

class BlockLocalSelfAttention(nn.Module):

    def __init__(self, block_size=128, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1):
        """
        This is expected to replace the vanilla Self Attention mechanism

        Compute block local attention with an optional global connection
        If compute_global_attention==True, the first query is connected to all keys and values 
            and all the other queries are connected to the first key and value (usually BOS token)
        
        WARNING: Causal is experimental especially for inference (cache)
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
        """
        Q, K, V: (batch, num_heads, sequence_length, hidden_size)
        mask:    (batch, 1, 1, sequence_length)
        """

        # Require Q K and V to be of the same size
        assert query_layer.size() == key_layer.size() == value_layer.size(), "Q, K, V have to be of the same size"

        n, h, t, d = query_layer.size()

        # Create mask if there is none
        # attention_mask: (batch, 1, 1, sequence_length)  (-inf for mask, 0 else)
        if attention_mask is None:
            attention_mask = torch.zeros(n, 1, 1, t, device=query_layer.device, dtype=query_layer.dtype)

        # If sequence is shorter than 2 blocks -> returns vanille self attention
        if t <= 2*self.block_size:
            return self.attention_product(query_layer, key_layer, value_layer, attention_mask)

        # Compute block local attention
        outputs = self.block_local_forward(query_layer, key_layer, value_layer, attention_mask)

        # Compute global attention
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

        # (batch, num_heads, sequence_length, hidden_size)
        n, h, t, d = query_layer.size()

        # Require to have sequence_length % block_size == 0
        extra_tokens = t % self.block_size

        # If sequence_length % block_size != 0, we pad
        if extra_tokens > 0:
            pad = (0, self.block_size - extra_tokens)
            query_layer = self.pad_inputs(query_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)
            key_layer = self.pad_inputs(key_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)
            value_layer = self.pad_inputs(value_layer.transpose(-1, -2), pad=pad).transpose(-1, -2)

            pad = (self.block_size - extra_tokens, self.block_size - extra_tokens) if self.is_causal else pad
            attention_mask = self.pad_inputs(attention_mask, pad=pad, value=torch.finfo(attention_mask.dtype).min)

        # If we compute global attention, we add a connection to the first token
        # A query is connected to : previous block, current block, next block, first token

        # We build K, V of sizes (batch, num_heads, num_blocks, block_size*3 (+1 if global), hidden_size)
        # We build the mask of size (batch, 1, num_blocks, 1, block_size*3 (+1 if global))
        key_layer = self.build_block_local_inputs(
            key_layer, 
            )

        value_layer = self.build_block_local_inputs(
            value_layer, 
            )
        
        attention_mask = self.build_block_local_inputs(
            attention_mask, 
            is_attn_mask=True
            ).transpose(-1, -2)

        # Expects (..., t, d) shape
        # Simple dot product attention between: 
        #   Q:      (batch, num_heads, num_blocks, block_size,        hidden_size)
        #   K, V:   (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        #   Mask:   (batch, 1,         num_blocks, 1,                 block_size*3 (+1))
        context_layer = self.attention(
                query_layer=self.chunk_to_blocks(query_layer), 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                )
        
        # We reshape and cut the sequence if we padded
        return context_layer.reshape(n, h, -1, d)[..., :t, :]
    
    def attention_product(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Apply the attention mask
        attention_scores = attention_scores + attention_mask
        del attention_mask

        # Normalize the attention scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer

    def causal_attention_product(self, query_layer, key_layer, value_layer, attention_mask=None, causal_shape=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Add causal mask
        causal_shape = (self.block_size, self.block_size) if causal_shape is None else causal_shape
        causal_mask = torch.tril(
            torch.ones(*causal_shape, device=attention_mask.device, dtype=attention_scores.dtype), 
            diagonal=-1
            ) 
        
        dtype_min = torch.tensor(
                    torch.finfo(attention_scores.dtype).min, device=attention_scores.device, dtype=attention_scores.dtype
                )

        causal_mask = self.pad_inputs(causal_mask.T * dtype_min, (attention_mask.size()[-1] - self.block_size, 0), value=0)

        attention_mask = attention_mask[..., -1:, :] + causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        attention_mask = torch.max(attention_mask, dtype_min)

        attention_scores = attention_scores + attention_mask

        del attention_mask
        del causal_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer

    def build_block_local_inputs(self, inputs, is_attn_mask=False):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        """

        # If we compute global attention, we add a connection to the first token 
        # A query is connected to : previous block, current block, next block, (first token)
        # Thus we split our sequences into overlapping blocks of size block_size*3 (+1)
        if self.compute_global_attention:
            if is_attn_mask:
                # We build the global mask
                global_inputs = torch.zeros(inputs.size()[0], 1, 1, 1, device=inputs.device, dtype=inputs.dtype)
                # We need to avoid a double connection to the first token in the first block
                # For this we mask the first token
                inputs[..., 0] = torch.finfo(inputs.dtype).min 
            else:
                # We extract the global key or value (first token)
                global_inputs = inputs[..., :1, :]

            # Returns (batch, num_heads, num_blocks, block_size*3 + 1, hidden_size)
            return self.concat_global_and_local_tokens(
                global_inputs, 
                self.reshape_to_block_local(inputs, is_attn_mask)
                )
        # Else returns (batch, num_heads, num_blocks, block_size*3, hidden_size)
        return self.reshape_to_block_local(inputs, is_attn_mask)

    def pad_inputs(self, inputs, pad, value=0):
        return torch.nn.functional.pad(inputs, pad=pad, value=value)

    def reshape_to_block_local(self, inputs, is_attn_mask=False):
        
        size, step = self.local_shapes
        s = (size - step) // 2

        # For shape consistency, we need to pad before reshaping
        # To get num_blocks of 3*block_size
        if is_attn_mask:
            pad_value = torch.finfo(inputs.dtype).min 
            inputs = inputs.transpose(-1, -2)
        else: 
            pad_value = 0

        inputs = torch.nn.functional.pad(
            inputs.transpose(-1, -2), 
            pad=(s, s),
            value=pad_value
            ).transpose(-1, -2)

        # Split into overlapping blocks
        inputs = inputs.unfold(-2, size=size, step=step).transpose(-1, -2)

        # Skip third block if causal
        if self.is_causal:
            return inputs[..., :size*2//3, :]

        return inputs

    def concat_global_and_local_tokens(self, global_inputs, inputs, dim=-2):
        """
        Concat together global and local tokens
        """
        n, h, b, t, d = inputs.size()
        global_inputs = global_inputs.unsqueeze(-3).expand(-1, -1, b, -1, -1)
        return torch.cat([global_inputs, inputs], dim=dim)

    def chunk_to_blocks(self, inputs):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, sequence_length//block_size, block_size, hidden_size)
        """
        t, d = inputs.size()[-2:]
        return inputs.reshape(*inputs.size()[:-2], t//self.block_size, -1, d)

