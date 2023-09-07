import torch
import torch.nn as nn
import math

class BaseLSGSelfAttention(nn.Module):

    def __init__(self, config=None, block_size=128, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1, preprocessing_function=None):
        """
        This is a substitute of vanilla Self Attention (bidirectionnal or causal)
        Doesn't work for Cross Attention because the local context is ambiguous to define in this case

        Compute block local attention with an optional global connection
        If compute_global_attention==True, the first query is connected to all keys and values 
            and all the other queries are connected to the first key and value (usually BOS token)
        
        WARNING: Causal is experimental especially for inference (we use full attention for generation)
        """
        super().__init__()

        self.config = config
        self.compute_global_attention = compute_global_attention
        self.block_size = block_size
        self.is_causal = is_causal
        self.dropout = nn.Dropout(attention_dropout_prob)
        self.preprocess = preprocessing_function if preprocessing_function is not None else self.preprocess

        # Shape of blocks
        self.local_shapes = (self.block_size*3, self.block_size)

        self.global_attention_product = self.attention_product
        self.block_causal_mask = None
        self.post_init()

    def post_init(self):
        pass 
    
    def preprocess(self, query_layer, key_layer, value_layer, attention_mask, **kwargs):
        return query_layer, key_layer, value_layer, attention_mask
    
    def build_triangular_mask(self, causal_shape, dtype, device):
        # Build triangular mask that works for torch.bfloat16
        mask = torch.full(causal_shape, torch.finfo(dtype).min, device=device, dtype=dtype)
        mask_cond = torch.arange(mask.size(-1), device=device, dtype=dtype)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        return mask
    
    def build_causal_mask(self, attention_mask, causal_shape, is_block_causal=False):
        device, dtype = attention_mask.device, attention_mask.dtype
        dtype_min = torch.tensor(
                        torch.finfo(dtype).min, device=device, dtype=dtype
                    )
        
        # Triangular mask (old behavior doesnt work for torch.bfloat16)
        # causal_mask = torch.ones(*causal_shape, device=attention_mask.device, dtype=attention_mask.dtype)
        # causal_mask = torch.tril(causal_mask, diagonal=-1).T       

        if is_block_causal:
            if self.block_causal_mask is None:
                self.block_causal_mask = self.build_triangular_mask(causal_shape, dtype, device)

            causal_mask = self.pad_inputs(
                inputs=self.block_causal_mask,# * dtype_min, 
                pad=(attention_mask.size()[-1] - self.block_size, 0), 
                value=0, 
                is_attn_mask=True
                )
            attention_mask = attention_mask + causal_mask[None, None, None, :, :]
        else:
            causal_mask = self.build_triangular_mask(causal_shape, dtype, device)
            attention_mask = attention_mask + (causal_mask)[None, None, :, :] 

        attention_mask = torch.max(attention_mask, dtype_min)
        return attention_mask
    
    def attention_product_for_generation(self, query_layer, key_layer, value_layer, attention_mask):
        # Should use the mask to extract the last 2*self.block_size tokens of each sequence for batch processing
        # But its a lot easier to compute full attention during generation instead 
        # May have a slight impact on generation
        # Can be easy to fix for batch_size = 1 since there is no padding in keys and values
        return self.attention_product(query_layer, key_layer, value_layer, attention_mask)
    
    def forward(
        self, 
        query_layer, 
        key_layer, 
        value_layer, 
        attention_mask=None, 
        **kwargs
        ):
        """
        Q, K, V: (batch, num_heads, sequence_length, hidden_size)
        mask:    (batch, 1, 1, sequence_length)
        """

        # Preprocessing function (default: do nothing)
        query_layer, key_layer, value_layer, attention_mask = self.preprocess(
                query_layer, 
                key_layer, 
                value_layer, 
                attention_mask, 
                **kwargs
                )
            
        n, h, t, d = query_layer.size()
        is_causal_mask = False

        # Check if we are generating
        if self.is_causal and not self.training:
            if t != key_layer.size()[-2]:
                # Should use the mask to extract the last 2*self.block_size tokens of each sequence for batch processing
                # But its a lot easier to compute full attention during generation instead 
                # Block causal attention is only used during the first step where query.size() == key.size()
                return self.attention_product_for_generation(query_layer, key_layer, value_layer, attention_mask)
                
        # Require Q K and V to be of the same size
        #assert query_layer.size() == key_layer.size() == value_layer.size(), "Q, K, V have to be of the same shape"

        # Create mask if there is none
        # attention_mask: (batch, 1, 1, sequence_length)  (-inf for mask, 0 else)
        if attention_mask is None:
            attention_mask = torch.zeros(n, 1, 1, t, device=query_layer.device, dtype=query_layer.dtype)
        else:
            assert len(attention_mask.size()) == 4, "Mask must have 4 dimensions, i.e (batch, 1, 1, sequence_length)"
            is_causal_mask = (attention_mask.size()[-2] == t and self.is_causal)

        # If sequence is shorter than 2 blocks -> return vanilla self attention
        if t <= 2*self.block_size:
            if not is_causal_mask and self.is_causal:
                attention_mask = self.build_causal_mask(attention_mask, causal_shape=(t, t))
            return self.attention_product(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask)
        
        # If mask is (batch, 1, seq_length, seq_length), extract the last row
        attention_mask = attention_mask[:, :1, -1:, :]

        # We need sequence_length % block_size == 0 to build blocks
        extra_tokens = t % self.block_size

        # If sequence_length % block_size != 0, we pad
        if extra_tokens > 0:
            query_layer = self.pad_inputs(query_layer, pad=(0, self.block_size - extra_tokens))
            # We pad keys, values and mask later

        return self._forward(query_layer, key_layer, value_layer, attention_mask)[..., :t, :]

    def _forward(self, query_layer, key_layer, value_layer, attention_mask):
        raise NotImplementedError
    
    def block_forward(query_layer, key_layer, value_layer, attention_mask, attention_scores=None):
        raise NotImplementedError 

    def block_reshape_inputs(inputs, is_attn_mask=False):
        raise NotImplementedError
    
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
    
    def bos_pooling_global_attention_product(self, query_layer, key_layer, value_layer, attention_mask=None, return_scores=False):
        
        # return attention_scores in some cases for bos_pooling
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        # Apply the attention mask
        attention_scores = attention_scores + attention_mask
        del attention_mask

        #if self.is_causal or not self.compute_global_attention:
        if return_scores:
            return (None, attention_scores)
        
        # Normalize the attention scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper
        context_layer = self.dropout(attention_probs) @ value_layer

        return (context_layer, attention_scores)
    
    def get_sparse_inputs(self, key_layer, value_layer, attention_mask, attention_scores=None):
        raise NotImplementedError
    
    def preprocess_sparse_inputs(self, key_layer, value_layer, attention_mask, attention_scores=None):
        
        # Pad the input to be divisible by block size
        n, h, t, d = key_layer.size()
        extra_tokens = t % self.block_size
        offset = self.block_size - extra_tokens if extra_tokens > 0 else 0
        if offset > 0:
            # For shape consistency, we need to pad before reshaping
            # To get num_blocks of 3*block_size
            key_layer = self.pad_inputs(
                key_layer, 
                pad=(0, offset),
                value=0,
                )
            
            value_layer = self.pad_inputs(
                value_layer, 
                pad=(0, offset),
                value=0,
                )
            
            attention_mask = self.pad_inputs(
                attention_mask.transpose(-1, -2), 
                pad=(0, offset),
                value=torch.finfo(attention_mask.dtype).min,
                ).transpose(-1, -2)
            
            if attention_scores is not None:
                attention_scores = self.pad_inputs(
                    attention_scores.transpose(-1, -2), 
                    pad=(0, offset),
                    value=torch.finfo(attention_mask.dtype).min,
                    ).transpose(-1, -2)
                
        return key_layer, value_layer, attention_mask, attention_scores
     
    def process_sparse_inputs(self, key_layer, value_layer, attention_mask, attention_scores=None):

        # Reshape to be divisible by block_size
        key_layer, value_layer, attention_mask, attention_scores = self.preprocess_sparse_inputs(
            key_layer, 
            value_layer, 
            attention_mask, 
            attention_scores
            )
        
        # No sparsity so we return everything
        if self.sparsity_factor == 1:
            return key_layer, value_layer, attention_mask, attention_scores

        # Select sparse token using a custom function
        return self.get_sparse_inputs(key_layer, value_layer, attention_mask, attention_scores)
    
    def process_global_inputs(self, key_layer, value_layer, attention_mask, global_idx=None):

        # Set a padding_value
        pad_value = torch.finfo(attention_mask.dtype).min

        # If we compute global attention, we add a connection to the first token 
        # A query is connected to : previous block, current block, next block, (first token)
        # Thus we split our sequences into overlapping blocks of size block_size*3 (+1)
        
        # We extract the global key, value, mask (first token)
        global_key_layer = key_layer[..., :1, :].clone()
        global_value_layer = value_layer[..., :1, :].clone()
        global_attention_mask = torch.zeros(*attention_mask.size()[:2], 1, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        # We need to avoid a double connection to the first token in the first block
        # For this we mask the first token (-inf) (NOTE: attn_mask is transposed)
        attention_mask[..., 0] = pad_value 
        return global_key_layer, global_value_layer, global_attention_mask, attention_mask

    def reshape_to_block_local(self, inputs, pad_value):

        # Reshape into overlapping blocks
        n, h, t, d = inputs.size()
        extra_tokens = t % self.block_size
        offset = self.block_size - extra_tokens if extra_tokens > 0 else 0

        size, step = self.local_shapes
        s = (size - step) // 2

        # For shape consistency, we need to pad before reshaping
        # To get num_blocks of 3*block_size
        inputs = self.pad_inputs(
            inputs, 
            pad=(s, s + offset),
            value=pad_value,
            )

        # Split into overlapping blocks
        inputs = inputs.unfold(-2, size=size, step=step).transpose(-1, -2)

        # Skip third block if causal
        if self.is_causal:
            return inputs[..., :size*2//3, :]

        return inputs
    
    def reshape_to_block_sparse(self, inputs, pad_value):
        
        # Reshape into overlapping blocks
        n, h, t, d = inputs.size()

        size, step = (self.block_size*3, self.block_size//self.sparsity_factor)
        s = (size - step) // 2

        # In case of odd case
        odd_offset = (step % 2)

        # For shape consistency, we need to pad before reshaping
        # To get num_blocks of 3*block_size
        inputs = self.pad_inputs(
            inputs, 
            pad=(s, s),
            value=pad_value,
            )
        # Make blocks
        inputs = inputs.unfold(-2, size=size, step=step).transpose(-1, -2)

        # Indexes for selection
        u = (size - self.block_size * 3 // self.sparsity_factor) // 2 + odd_offset
        u_ = u + odd_offset

        if self.is_causal:
            return inputs[..., u-self.block_size:u, :]
        return torch.cat([inputs[..., u-self.block_size:u, :], inputs[..., -u_:-u_+self.block_size, :]], dim=-2)

    def chunk_to_blocks(self, inputs):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, sequence_length//block_size, block_size, hidden_size)
        """
        n, h, t, d = inputs.size()
        return inputs.reshape(n, h, t//self.block_size, -1, d)

    def pad_inputs(self, inputs, pad, value=0, is_attn_mask=False):
        if not is_attn_mask:
            return torch.nn.functional.pad(inputs.transpose(-1, -2), pad=pad, value=value).transpose(-1, -2)
        return torch.nn.functional.pad(inputs, pad=pad, value=value)

    def concat_global_and_local_tokens(self, global_inputs, inputs, dim=-2):
        """
        Concat together global and local tokens
        """
        n, h, b, t, d = inputs.size()
        global_inputs = global_inputs.unsqueeze(-3).expand(-1, -1, b, -1, -1)
        return torch.cat([global_inputs, inputs], dim=dim)

    def merge_tokens(self, inputs, sparse_inputs, global_inputs=None, dim=-2):
        """
        Concat together local sparse and global tokens
        """
        n, h, b, t, d = inputs.size()
        h_ = sparse_inputs.size()[1]
        if h != h_:
            inputs = inputs.expand(-1, max(h, h_), -1, -1, -1)

        if global_inputs is not None:
            global_inputs = global_inputs.unsqueeze(-3).expand(-1, max(h, h_), b, -1, -1)
            return torch.cat([global_inputs, sparse_inputs, inputs], dim=dim)
        return torch.cat([sparse_inputs, inputs], dim=dim)

    def concat_global_and_local_tokens(self, global_inputs, inputs, dim=-2):
        """
        Concat together global and local tokens
        """
        n, h, b, t, d = inputs.size()
        global_inputs = global_inputs.unsqueeze(-3).expand(-1, -1, b, -1, -1)
        return torch.cat([global_inputs, inputs], dim=dim)
    

class BlockLocalSelfAttention(BaseLSGSelfAttention):

    def __init__(self, config=None, block_size=128, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1, preprocessing_function=None):
        """
        This is a substitute of vanilla Self Attention (bidirectionnal or causal)
        Doesn't work for Cross Attention because the local context is ambiguous to define in this case

        Compute block local attention with an optional global connection
        If compute_global_attention==True, the first query is connected to all keys and values 
            and all the other queries are connected to the first key and value (usually BOS token)
        
        WARNING: Causal is experimental especially for inference (we use full attention for generation)
        """
        super().__init__(config, block_size, compute_global_attention, is_causal, attention_dropout_prob, preprocessing_function)

    def _forward(self, query_layer, key_layer, value_layer, attention_mask):

        # If we compute global attention, we add a connection to the first token
        # A query is then connected to : previous block, current block, next block, first token
        n, h, t, d = query_layer.size()

        global_query_layer = None
        global_key_layer, global_value_layer, global_attention_mask = None, None, None
        if self.compute_global_attention:
            if not self.is_causal:
                global_query_layer = self.global_attention_product(query_layer[..., :1, :], key_layer, value_layer, attention_mask)
            global_key_layer, global_value_layer, global_attention_mask, attention_mask = self.process_global_inputs(
                key_layer, 
                value_layer, 
                attention_mask
                )
            
        # We build K, V of sizes: (batch, num_heads, num_blocks, block_size*3 (+1 if global), hidden_size)
        # We build the mask of size: (batch, 1, num_blocks, 1, block_size*3 (+1 if global))
        key_layer = self.block_reshape_inputs(key_layer, global_key_layer)
        value_layer = self.block_reshape_inputs(value_layer, global_value_layer)
        
        # Need to transpose attention_mask to follow K and V format
        attention_mask = self.block_reshape_inputs(
            attention_mask.transpose(-1, -2), 
            global_attention_mask.transpose(-1, -2) if global_attention_mask is not None else None,
            is_attn_mask=True
            ).transpose(-1, -2)

        # Prepare causal_mask for dot product
        if self.is_causal:
            attention_mask = self.build_causal_mask(
                attention_mask, 
                causal_shape=(self.block_size, self.block_size), 
                is_block_causal=True
                )

        # Expect (..., t, d) shapes
        # Simple dot product attention between: 
        #   Q:      (batch, num_heads, num_blocks, block_size,        hidden_size)
        #   K, V:   (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        #   Mask:   (batch, 1,         num_blocks, 1,                 block_size*3 (+1))
        context_layer = self.attention_product(
                query_layer=self.chunk_to_blocks(query_layer), 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                ).reshape(n, h, -1, d)
        
        if global_query_layer is not None:
            context_layer[..., :1, :] = global_query_layer 
        return context_layer

    def block_reshape_inputs(self, inputs, global_inputs=None, is_attn_mask=False):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        """
        n, h, t, d = inputs.size()

        # Set a padding_value
        pad_value = torch.finfo(inputs.dtype).min if is_attn_mask else 0

        # If we compute global attention, we add a connection to the first token 
        # A query is connected to : previous block, current block, next block, (first token)
        # Thus we split our sequences into overlapping blocks of size block_size*3 (+1)
        if global_inputs is not None:
            # Return (batch, num_heads, num_blocks, block_size*3 + 1, hidden_size)
            return self.concat_global_and_local_tokens(
                global_inputs, 
                self.reshape_to_block_local(inputs, pad_value)
                )
        # Else return (batch, num_heads, num_blocks, block_size*3, hidden_size)
        return self.reshape_to_block_local(inputs, pad_value)


class LSGSelfAttention(BaseLSGSelfAttention):

    def __init__(self, config=None, block_size=128, sparsity_factor=8, sparsity_type=None, compute_global_attention=True, is_causal=False, attention_dropout_prob=0.1, preprocessing_function=None):
        """
        This is a substitute of vanilla Self Attention (bidirectionnal or causal)
        Doesn't work for Cross Attention because the local context is ambiguous to define in this case

        Compute block local attention with an optional global connection
        If compute_global_attention==True, the first query is connected to all keys and values 
            and all the other queries are connected to the first key and value (usually BOS token)
        
        WARNING: Causal is experimental especially for inference (we use full attention for generation)
        """
        super().__init__(config, block_size, compute_global_attention, is_causal, attention_dropout_prob, preprocessing_function)

        self.sparsity_factor = sparsity_factor
        self.sparsity_type = sparsity_type
        
        sparse_function_dict = {
            "bos_pooling": self.get_sparse_tokens_with_bos_pooling, 
            "norm": self.get_sparse_tokens_with_norm, 
            "pooling": self.get_sparse_tokens_with_pooling,
            "stride": self.get_sparse_tokens_with_stride,
            "block_stride": self.get_sparse_tokens_with_block_stride,
            }
        
        if sparsity_type is not None:
            if not sparsity_type in sparse_function_dict.keys():
                self.sparsity_type = None

        if self.sparsity_type is not None:
            self.get_sparse_inputs = sparse_function_dict[sparsity_type]

    def _forward(self, query_layer, key_layer, value_layer, attention_mask):

        # If we compute global attention, we add a connection to the first token
        # A query is then connected to : previous block, current block, next block, first token
        n, h, t, d = query_layer.size()

        # Get global tokens and compute global attention
        global_query_layer, attention_scores = None, None
        global_key_layer, global_value_layer, global_attention_mask = None, None, None

        # Special case with bos_pooling
        if self.sparsity_type == "bos_pooling":
            (global_query_layer, attention_scores) = self.bos_pooling_global_attention_product(
                query_layer[..., :1, :], 
                key_layer, 
                value_layer, 
                attention_mask,
                return_scores=self.is_causal
                )
        if self.compute_global_attention:
            if not self.is_causal and global_query_layer is None:
                global_query_layer = self.global_attention_product(query_layer[..., :1, :], key_layer, value_layer, attention_mask)

            # Get global tokens
            global_key_layer, global_value_layer, global_attention_mask, attention_mask = self.process_global_inputs(
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask
                )
        
        # Get sparse token
        sparse_key_layer, sparse_value_layer, sparse_attention_mask = None, None, None
        if self.sparsity_type is not None:
            sparse_key_layer, sparse_value_layer, sparse_attention_mask = self.process_sparse_inputs(
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask, 
                attention_scores=attention_scores
                )

        # We build K, V of sizes: (batch, num_heads, num_blocks, block_size*3 (+1 if global), hidden_size)
        # We build the mask of size: (batch, 1, num_blocks, 1, block_size*3 (+1 if global))
        key_layer = self.block_reshape_inputs(key_layer, sparse_key_layer, global_key_layer)
        value_layer = self.block_reshape_inputs(value_layer, sparse_value_layer, global_value_layer)
        
        # Need to transpose attention_mask to follow K and V format
        attention_mask = self.block_reshape_inputs(
            attention_mask.transpose(-1, -2), 
            sparse_attention_mask.transpose(-1, -2) if sparse_attention_mask is not None else None,
            global_attention_mask.transpose(-1, -2) if global_attention_mask is not None else None,
            is_attn_mask=True
            ).transpose(-1, -2)

        # Prepare causal_mask for dot product
        if self.is_causal:
            attention_mask = self.build_causal_mask(
                attention_mask, 
                causal_shape=(self.block_size, self.block_size), 
                is_block_causal=True
                )

        # Expect (..., t, d) shapes
        # Simple dot product attention between: 
        #   Q:      (batch, num_heads, num_blocks, block_size,        hidden_size)
        #   K, V:   (batch, num_heads, num_blocks, block_size*5 (+1), hidden_size)
        #   Mask:   (batch, 1,         num_blocks, 1,                 block_size*5 (+1))
        context_layer = self.attention_product(
                query_layer=self.chunk_to_blocks(query_layer), 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                ).reshape(n, h, -1, d)
        
        if global_query_layer is not None:
            context_layer[..., :1, :] = global_query_layer 
        return context_layer

    def get_sparse_tokens_with_bos_pooling(self, key_layer, value_layer, attention_mask, attention_scores):

        n, h, t, d = key_layer.size()
        
        key_layer = key_layer.reshape(n, h, -1, self.sparsity_factor, d)
        value_layer = value_layer.reshape(n, h, -1, self.sparsity_factor, d)
        attention_mask = attention_mask.reshape(n, 1, -1, 1, self.sparsity_factor)

        # Fix for multiqueries where keys and values have 1 head and scores h heads
        attention_scores = attention_scores.reshape(*attention_scores.size()[:2], -1, 1, self.sparsity_factor)

        n, h, b, t, d = key_layer.size()

        attention_scores = attention_scores.reshape(*attention_scores.size()[:2], -1, 1, self.sparsity_factor)

        attention_scores = torch.softmax(attention_scores, dim=-1)
        key_layer = attention_scores @ key_layer
        value_layer = attention_scores @ value_layer
        attention_mask = attention_mask.mean(dim=-1)
        attention_mask[attention_mask != torch.finfo(attention_mask.dtype).min] = 0

        return key_layer.reshape(*attention_scores.size()[:2], -1, d), value_layer.reshape(*attention_scores.size()[:2], -1, d), attention_mask.reshape(n, 1, 1, -1)
    
    def get_sparse_tokens_with_norm(self, key_layer, value_layer, attention_mask, attention_scores=None):

        n, h, t, d = key_layer.size()
        
        key_layer = key_layer.reshape(n, h, -1, self.block_size, d)
        value_layer = value_layer.reshape(n, h, -1, self.block_size, d)
        attention_mask = attention_mask.reshape(n, 1, -1, 1, self.block_size)

        n, h, b, t, d = key_layer.size()

        with torch.no_grad():

            key_norm = key_layer.detach().norm(dim=-1, keepdim=True)
            key_norm = key_norm * ~attention_mask.transpose(-1, -2).bool()
            idx = torch.topk(key_norm, k=t//self.sparsity_factor, dim=-2, largest=True, sorted=False)[1]
            del key_norm

        key_layer = key_layer.gather(dim=-2, index=idx.expand(-1, -1, -1, -1, d))
        value_layer = value_layer.gather(dim=-2, index=idx.expand(-1, -1, -1, -1, d))
        attention_mask = attention_mask.expand(-1, h, -1, -1, -1).transpose(-1, -2).gather(dim=-2, index=idx).transpose(-1, -2)

        return key_layer.reshape(n, h, -1, d), value_layer.reshape(n, h, -1, d), attention_mask.reshape(*attention_mask.size()[:2], 1, -1)
    
    def get_sparse_tokens_with_pooling(self, key_layer, value_layer, attention_mask, attention_scores=None):

        n, h, t, d = key_layer.size()
        
        key_layer = key_layer.reshape(n, h, -1, self.sparsity_factor, d)
        value_layer = value_layer.reshape(n, h, -1, self.sparsity_factor, d)
        mask = ~attention_mask.reshape(n, 1, -1, 1, self.sparsity_factor).transpose(-1, -2).bool()
        mask = mask.to(attention_mask.dtype)

        n, h, b, t, d = key_layer.size()

        key_layer = key_layer * mask
        value_layer = value_layer * mask

        mask = mask.sum(dim=-2)
        key_layer = key_layer.sum(dim=-2) / (mask + 1e-6)
        value_layer = value_layer.sum(dim=-2) / (mask + 1e-6)
        attention_mask = attention_mask.reshape(n, 1, -1, 1, self.sparsity_factor).mean(dim=-1)
        attention_mask[attention_mask != torch.finfo(attention_mask.dtype).min] = 0

        return key_layer.reshape(n, h, -1, d), value_layer.reshape(n, h, -1, d), attention_mask.transpose(-1, -2)
    
    def get_sparse_tokens_with_stride(self, key_layer, value_layer, attention_mask, attention_scores=None):

        n, h, t, d = key_layer.size()
        sparse_idx = torch.arange(t // self.sparsity_factor, device=key_layer.device) * self.sparsity_factor
        sparse_idx = sparse_idx.reshape(1, 1, -1, 1) + (torch.arange(h, device=key_layer.device) % self.sparsity_factor).reshape(1, h, 1, 1)
        sparse_idx = sparse_idx.expand(n, h, -1, 1)

        key_layer = key_layer.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        value_layer = value_layer.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        attention_mask = attention_mask.expand(-1, h, -1, -1).transpose(-1, -2).gather(dim=-2, index=sparse_idx).transpose(-1, -2)

        return key_layer, value_layer, attention_mask

    def get_sparse_tokens_with_block_stride(self, key_layer, value_layer, attention_mask, attention_scores=None):

        n, h, t, d = key_layer.size()

        t, b = self.block_size, t // self.block_size
        sparse_idx = torch.arange(t // self.sparsity_factor, device=key_layer.device)
        sparse_idx = sparse_idx.reshape(1, 1, 1, -1, 1) + torch.arange(h, device=key_layer.device).reshape(1, h, 1, 1, 1) * (t // self.sparsity_factor)
        sparse_idx = (sparse_idx % t) 
        sparse_idx = sparse_idx + torch.arange(b, device=key_layer.device).reshape(1, 1, -1, 1, 1) * t
        sparse_idx = sparse_idx.reshape(1, h, -1, 1).expand(n, h, -1, 1)

        key_layer = key_layer.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        value_layer = value_layer.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        attention_mask = attention_mask.expand(-1, h, -1, -1).transpose(-1, -2).gather(dim=-2, index=sparse_idx).transpose(-1, -2)

        return key_layer, value_layer, attention_mask
    
    def block_reshape_inputs(self, inputs, sparse_inputs=None, global_inputs=None, is_attn_mask=False):
        """
        Transforms an input: 
            (batch, num_heads, sequence_length, hidden_size)
        to
            (batch, num_heads, num_blocks, block_size*3 (+1), hidden_size)
        """
        n, h, t, d = inputs.size()

        # Set a padding_value
        pad_value = torch.finfo(inputs.dtype).min if is_attn_mask else 0
        
        # If we compute global attention, we add a connection to the first token 
        # A query is connected to : previous block, current block, next block, (first token)
        # Thus we split our sequences into overlapping blocks of size block_size*3 (+1)

        # Return (batch, num_heads, num_blocks, block_size*(3 + 2) + 1, hidden_size)
        if sparse_inputs is not None:
            return self.merge_tokens(
                self.reshape_to_block_local(inputs, pad_value),
                self.reshape_to_block_sparse(sparse_inputs, pad_value),
                global_inputs, 
                )
        if global_inputs is not None:
            # Return (batch, num_heads, num_blocks, block_size*3 + 1, hidden_size)
            return self.concat_global_and_local_tokens(
                global_inputs, 
                self.reshape_to_block_local(inputs, pad_value)
                )
        # Else return (batch, num_heads, num_blocks, block_size*3, hidden_size)
        return self.reshape_to_block_local(inputs, pad_value)
    
    def merge_tokens(self, inputs, sparse_inputs, global_inputs=None, dim=-2):
        """
        Concat together global and local tokens
        """
        n, h, b, t, d = inputs.size()
        h_ = sparse_inputs.size()[1]
        if h != h_:
            inputs = inputs.expand(-1, max(h, h_), -1, -1, -1)

        if global_inputs is not None:
            global_inputs = global_inputs.unsqueeze(-3).expand(-1, max(h, h_), b, -1, -1)
            return torch.cat([global_inputs, sparse_inputs, inputs], dim=dim)
        
        return torch.cat([sparse_inputs, inputs], dim=dim)

