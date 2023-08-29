import torch
import torch.nn as nn
import math

class BlockLocalSelfAttention(nn.Module):

    def __init__(self, block_size=128, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1):
        """
        This is expected to replace the vanilla Self Attention mechanism
        Should be compatible with flash-attention with various reshapes

        Compute block local attention with an optional global connection
        If compute_global_attention==True, the first query is connected to all keys and values 
            and all the other queries are connected to the first key and value (usually BOS token)
        
        WARNING: Causal is experimental especially for inference (we use full attention for generation)
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

        n, h, t, d = query_layer.size()

        # Check if we are generating
        if self.is_causal and not self.training:
            if t != key_layer.size()[-2]:
                # Should use the mask to extract the last 2*self.block_size tokens of each sequence for batch processing
                # But its a lot easier to compute full attention during generation instead 
                # Block causal attention is only used during the first step where query.size() == key.size()
                return self.attention_product_for_generation(query_layer, key_layer, value_layer, attention_mask)
                
        # Require Q K and V to be of the same size
        assert query_layer.size() == key_layer.size() == value_layer.size(), "Q, K, V have to be of the same size"

        # Create mask if there is none
        # attention_mask: (batch, 1, 1, sequence_length)  (-inf for mask, 0 else)
        if attention_mask is None:
            attention_mask = torch.zeros(n, 1, 1, t, device=query_layer.device, dtype=query_layer.dtype)

        # If sequence is shorter than 2 blocks -> return vanilla self attention
        if t <= 2*self.block_size:
            if self.is_causal:
                return self.causal_attention_product(
                    query_layer, 
                    key_layer, 
                    value_layer, 
                    attention_mask, 
                    causal_shape=(t, t), 
                    is_block_causal=False
                    )
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

        # We need sequence_length % block_size == 0 to build blocks
        extra_tokens = t % self.block_size

        # If sequence_length % block_size != 0, we pad
        if extra_tokens > 0:
            query_layer = self.pad_inputs(query_layer, pad=(0, self.block_size - extra_tokens))
            # We pad keys, values and mask later, when we build block_local_inputs

        # If we compute global attention, we add a connection to the first token
        # A query is then connected to : previous block, current block, next block, first token

        # We build K, V of sizes: (batch, num_heads, num_blocks, block_size*3 (+1 if global), hidden_size)
        # We build the mask of size: (batch, 1, num_blocks, 1, block_size*3 (+1 if global))
        key_layer = self.build_block_local_inputs(key_layer)
        value_layer = self.build_block_local_inputs(value_layer)
        
        # Need to transpose attention_mask to follow K and V format
        attention_mask = self.build_block_local_inputs(
            attention_mask.transpose(-1, -2), 
            is_attn_mask=True
            ).transpose(-1, -2)

        # Expect (..., t, d) shapes
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
    
    def attention_product_for_generation(self, query_layer, key_layer, value_layer, attention_mask):
        # Should use the mask to extract the last 2*self.block_size tokens of each sequence for batch processing
        # But its a lot easier to compute full attention during generation instead 
        # May have a slight impact on generation
        # Can be easy to fix for batch_size = 1 since there is no padding in keys and values
        return self.attention_product(query_layer, key_layer, value_layer, attention_mask)
    
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

    def causal_attention_product(self, query_layer, key_layer, value_layer, attention_mask=None, causal_shape=None, is_block_causal=True):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Add causal mask
        causal_shape = (self.block_size, self.block_size) if causal_shape is None else causal_shape
        attention_mask = self.build_causal_mask(attention_mask, causal_shape, is_block_causal)

        attention_scores = attention_scores + attention_mask

        del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer

    def build_causal_mask(self, attention_mask, causal_shape, is_block_causal):
        dtype_min = torch.tensor(
                        torch.finfo(attention_mask.dtype).min, device=attention_mask.device, dtype=attention_mask.dtype
                    )
        
        # Triangular mask
        causal_mask = torch.ones(*causal_shape, device=attention_mask.device, dtype=attention_mask.dtype)
        causal_mask = torch.tril(causal_mask, diagonal=-1).T
        
        if is_block_causal:
            causal_mask = self.pad_inputs(
                inputs=causal_mask * dtype_min, 
                pad=(attention_mask.size()[-1] - self.block_size, 0), 
                value=0, 
                is_attn_mask=True
                )
            attention_mask = attention_mask[..., -1:, :] + causal_mask[None, None, None, :, :]
        else:
            attention_mask = attention_mask + (causal_mask * dtype_min)[None, None, :, :] 

        attention_mask = torch.max(attention_mask, dtype_min)
        return attention_mask
    
    def build_block_local_inputs(self, inputs, is_attn_mask=False):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        """

        # Set a padding_value
        pad_value = torch.finfo(inputs.dtype).min if is_attn_mask else 0

        # If we compute global attention, we add a connection to the first token 
        # A query is connected to : previous block, current block, next block, (first token)
        # Thus we split our sequences into overlapping blocks of size block_size*3 (+1)
        if self.compute_global_attention:
            # We extract the global key, value, mask (first token)
            global_inputs = inputs[..., :1, :]
            if is_attn_mask:
                # We need to avoid a double connection to the first token in the first block
                # For this we mask the first token (-inf) (NOTE: attn_mask is transposed)
                inputs[..., 0, :] = pad_value 

            # Return (batch, num_heads, num_blocks, block_size*3 + 1, hidden_size)
            return self.concat_global_and_local_tokens(
                global_inputs, 
                self.reshape_to_block_local(inputs, pad_value)
                )
        # Else return (batch, num_heads, num_blocks, block_size*3, hidden_size)
        return self.reshape_to_block_local(inputs, pad_value)

    def reshape_to_block_local(self, inputs, pad_value):
        
        n, h, t, d = inputs.size()
        extra_tokens = t % self.block_size
        size, step = self.local_shapes
        s = (size - step) // 2

        # For shape consistency, we need to pad before reshaping
        # To get num_blocks of 3*block_size
        inputs = self.pad_inputs(
            inputs, 
            pad=(s, s + self.block_size - extra_tokens),
            value=pad_value,
            )

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

    def pad_inputs(self, inputs, pad, value=0, is_attn_mask=False):
        if not is_attn_mask:
            return torch.nn.functional.pad(inputs.transpose(-1, -2), pad=pad, value=value).transpose(-1, -2)
        return torch.nn.functional.pad(inputs, pad=pad, value=value)