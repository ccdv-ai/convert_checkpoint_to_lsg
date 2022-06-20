from logging import warn
from transformers.models.distilbert.modeling_distilbert import *
import torch
import torch.nn as nn
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
import sys


AUTO_MAP = {
        "AutoModel": "modeling_lsg_distilbert.LSGDistilBertModel",
        "AutoModelForMaskedLM": "modeling_lsg_distilbert.LSGDistilBertForMaskedLM",
        "AutoModelForMultipleChoice": "modeling_lsg_distilbert.LSGDistilBertForMultipleChoice",
        "AutoModelForQuestionAnswering": "modeling_lsg_distilbert.LSGDistilBertForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_lsg_distilbert.LSGDistilBertForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_lsg_distilbert.LSGDistilBertForTokenClassification"
    }


class LSGDistilBertConfig(DistilBertConfig):

    base_model_prefix = "lsg"
    model_type = "distilbert"

    def __init__(
        self,
        adaptive=True,
        base_model_prefix="lsg",
        block_size=128,
        lsh_num_pre_rounds=1,
        mask_first_token=False,
        num_global_tokens=1,
        pool_with_global=True,
        sparse_block_size=128,
        sparsity_factor=2,
        sparsity_type="norm",
        **kwargs
        ):
        """Constructs LSGDistilBertConfig."""
        super().__init__(**kwargs)

        self.adaptive = adaptive
        self.auto_map = AUTO_MAP
        self.base_model_prefix = base_model_prefix
        self.block_size = block_size
        self.lsh_num_pre_rounds = lsh_num_pre_rounds
        self.mask_first_token = mask_first_token
        self.num_global_tokens = num_global_tokens
        self.pool_with_global = pool_with_global
        self.sparse_block_size = sparse_block_size
        self.sparsity_factor = sparsity_factor
        self.sparsity_type = sparsity_type
        
        if sparsity_type not in [None, "none", "norm", "lsh", "pooling", "stride", "block_stride"]:
            logger.warning(
                "[WARNING CONFIG]: sparsity_mode not in [None, 'none', 'norm', 'lsh', 'pooling', 'stride', 'block_stride'], setting sparsity_type=None, computation will skip sparse attention")
            self.sparsity_type = None

        if self.sparsity_type in ["stride", "block_stride"]:
            if self.sparsity_factor > self.encoder_attention_heads:
                logger.warning(
                "[WARNING CONFIG]: sparsity_factor > encoder_attention_heads is not recommended for stride/block_stride sparsity"
            )

        if self.num_global_tokens < 1:
            logger.warning(
                "[WARNING CONFIG]: num_global_tokens < 1 is not compatible, setting num_global_tokens=1"
            )
            self.num_global_tokens = 1
        elif self.num_global_tokens > 512:
            logger.warning(
                "[WARNING CONFIG]: num_global_tokens > 512 is not compatible, setting num_global_tokens=512"
            )
            self.num_global_tokens = 512
        
        if self.sparsity_factor > 0:
            assert self.block_size % self.sparsity_factor == 0, "[ERROR CONFIG]: block_size must be divisible by sparsity_factor"
            assert self.block_size//self.sparsity_factor >= 1, "[ERROR CONFIG]: make sure block_size >= sparsity_factor"
        

class LSGEmbeddings(Embeddings):

    def __init__(self, config):

        super().__init__(config)
        self.num_global_tokens = config.num_global_tokens

        # Hardcoded but partially trained
        self.global_embeddings = nn.Embedding(512, embedding_dim=config.dim, )

        self.block_size = config.block_size

    def forward(self, input_ids, inputs_embeds=None):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        bs, seq_length = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(bs, seq_length)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids) if input_ids is not None else inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        word_embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)

        #if self.num_global_tokens < 0:
        n, t, d = word_embeddings.size()
        
        # Add global_tokens
        indexes = torch.arange(self.num_global_tokens, device=word_embeddings.device).reshape(1, -1)
        global_embeddings = self.global_embeddings(indexes) 
        word_embeddings = torch.cat([global_embeddings.expand(n, -1, d), word_embeddings], dim=-2)

        word_embeddings = self.LayerNorm(word_embeddings)  # (bs, max_seq_length, dim)
        word_embeddings = self.dropout(word_embeddings)  # (bs, max_seq_length, dim)
        return word_embeddings


class BaseSelfAttention(nn.Module):
    
    def init_modules(self, config):
        if config.dim % config.n_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim, config.n_heads)
            )

        self.n_heads = config.n_heads
        self.attention_head_size = int(config.dim / config.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.n_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_output(self, context_layer):
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_layer_shape)

    def project_QKV(self, hidden_states):

        query_layer = self.transpose_for_scores(self.q_lin(hidden_states))
        key_layer = self.transpose_for_scores(self.k_lin(hidden_states))
        value_layer = self.transpose_for_scores(self.v_lin(hidden_states))
        return query_layer, key_layer, value_layer


class BaseAttentionProduct(nn.Module):

    def __init__(self, config):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask
            del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer


class CausalAttentionProduct(nn.Module):

    def __init__(self, config):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_dropout)
        self.block_size = config.block_size

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None, causal_shape=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

            # Add causal mask
            causal_shape = (self.block_size, self.block_size) if causal_shape is None else causal_shape
            causal_mask = torch.tril(torch.ones(*causal_shape, device=attention_mask.device), diagonal=-1).T * (-10000)
            attention_scores[..., -causal_shape[0]:, -causal_shape[1]:] = causal_mask

            del attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        context_layer = self.dropout(attention_probs) @ value_layer

        return context_layer


class LSGAttentionProduct(nn.Module):

    def __init__(self, config, block_size=None, sparse_block_size=None, sparsity_factor=4, is_causal=False):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.block_size = block_size
        self.sparse_block_size = sparse_block_size
        self.sparsity_factor = sparsity_factor
        self.is_causal = is_causal

        if self.block_size is None:
            self.block_size = config.block_size

        if self.sparse_block_size is None:
            self.sparse_block_size = config.sparse_block_size

        # Shape of blocks
        self.local_shapes = (self.block_size*3, self.block_size)
        if self.sparse_block_size and self.sparsity_factor > 0:
            self.sparse_shapes = (self.sparse_block_size*3, self.block_size//self.sparsity_factor)

        if is_causal:
            self.attention = CausalAttentionProduct(config)
        else:
            self.attention = BaseAttentionProduct(config)
        
    def build_lsg_inputs(self, hidden_states, sparse_hidden_states, global_hidden_states, is_attn_mask=False):
        
        # Build local tokens
        local_hidden_states = self.reshape_to_local_block(hidden_states, is_attn_mask)
        del hidden_states

        # Build sparse tokens
        if sparse_hidden_states is not None:
            sparse_hidden_states = self.reshape_to_sparse_block(sparse_hidden_states, is_attn_mask)
        
        return self.cat_global_sparse_local_tokens(global_hidden_states, sparse_hidden_states, local_hidden_states)

    def forward(
        self, 
        query_layer, 
        key_layer, 
        value_layer, 
        attention_mask=None, 
        sparse_key=None,
        sparse_value=None, 
        sparse_mask=None, 
        global_key=None, 
        global_value=None, 
        global_mask=None
        ):

        # Input batch, heads, length, dim
        n, h, t, d = query_layer.size()
        n_blocks = t // self.block_size
        assert t % self.block_size == 0

        key_layer = self.build_lsg_inputs(
            key_layer, 
            sparse_key, 
            global_key
            )
        del sparse_key
        del global_key

        value_layer = self.build_lsg_inputs(
            value_layer, 
            sparse_value, 
            global_value
            )
        del sparse_value
        del global_value

        attention_mask = self.build_lsg_inputs(
            attention_mask, 
            sparse_mask, 
            global_mask.transpose(-1, -2), 
            is_attn_mask=True
            ).transpose(-1, -2)
        del sparse_mask
        del global_mask

        # expect (..., t, d) shape
        # Compute attention
        context_layer = self.attention(
                query_layer=self.chunk(query_layer, n_blocks), 
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask
                )
                
        return context_layer.reshape(n, h, -1, d)
    
    def reshape_to_local_block(self, hidden_states, is_attn_mask=False):
        
        size, step = self.local_shapes
        s = (size - step) // 2

        # Pad before block reshaping
        if is_attn_mask:
            pad_value = -10000  
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

    def reshape_to_sparse_block(self, hidden_states, is_attn_mask=False):
        
        size, step = self.sparse_shapes

        # In case of odd case
        odd_offset = (step % 2)

        # n, h, t, d*2 + 1
        size = size*2 
        s = (size - step) // 2 + odd_offset

        # Pad before block reshaping
        if is_attn_mask:
            pad_value = -10000  
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

        # Fix case where block_size == sparsify_factor
        if odd_offset: 
            hidden_states = hidden_states[..., :-1, :, :]

        # Indexes for selection
        u = (size - self.block_size * 3 // self.sparsity_factor) // 2 + odd_offset
        s = self.sparse_block_size

        # Skip right block if causal
        if self.is_causal:
            return hidden_states[..., u-s:u, :]

        u_ = u + odd_offset
        return torch.cat([hidden_states[..., u-s:u, :], hidden_states[..., -u_:-u_+s, :]], dim=-2)

    def cat_global_sparse_local_tokens(self, x_global, x_sparse=None, x_local=None, dim=-2):

        n, h, b, t, d = x_local.size()
        x_global = x_global.unsqueeze(-3).expand(-1, -1, b, -1, -1)
        if x_sparse is not None:
            return torch.cat([x_global, x_sparse, x_local], dim=dim)
        return torch.cat([x_global, x_local], dim=dim)

    def chunk(self, x, n_blocks):

        t, d = x.size()[-2:]
        return x.reshape(*x.size()[:-2], n_blocks, -1, d)


class LSGSelfAttention(BaseSelfAttention):
    '''
    Compute local attention with overlapping blocs
    Use global attention for tokens with highest norm
    '''
    def __init__(self, config):
        super().__init__()

        self.init_modules(config)

        self.block_size = config.block_size
        self.sparse_block_size = config.sparse_block_size
        self.num_global_tokens = config.num_global_tokens
        self.sparsity_factor = config.sparsity_factor
        self.is_causal = config.is_decoder
        self.is_decoder = config.is_decoder

        self.attention = LSGAttentionProduct(
            config, 
            block_size=config.block_size, 
            sparse_block_size=config.sparse_block_size, 
            sparsity_factor=self.sparsity_factor, 
            is_causal=self.is_causal
            )

        if self.is_causal:
            self.causal_attention = CausalAttentionProduct(config)
        self.full_attention = BaseAttentionProduct(config)

        sparse_functions = {
            "norm": self.get_sparse_tokens_with_norm, 
            "pooling": self.get_sparse_tokens_with_pooling,
            "lsh": self.get_sparse_tokens_with_lsh,
            "stride": self.get_sparse_tokens_with_stride,
            "block_stride": self.get_sparse_tokens_with_block_stride,
            }
        
        self.sparsity_type = config.sparsity_type
        self.get_sparse_elements = sparse_functions.get(self.sparsity_type, lambda x, y, z: (None, None, None))
            
        if config.sparsity_type == "lsh":
            self.lsh_num_pre_rounds = config.lsh_num_pre_rounds

    def get_sparse_tokens_with_norm(self, keys, values, mask):
        
        if self.sparsity_factor == 1:
            return keys, values, mask.expand(-1, keys.size()[1], -1, -1)

        with torch.no_grad():

            block_size = min(self.block_size, self.sparse_block_size)
            key_norm = keys.detach().norm(dim=-1, keepdim=True)
            key_norm = key_norm * ~mask.transpose(-1, -2).bool()
            key_norm = self.chunk(key_norm, block_size)

            n, h, b, t, d = key_norm.size()
            
            idx = key_norm.argsort(dim=-2) 
            del key_norm
            idx += (torch.arange(b, device=keys.device)*t).reshape(1, 1, b, 1, 1)

            split = (t - block_size // self.sparsity_factor, block_size // self.sparsity_factor)
            sparse_idx = idx.split(split, -2)[-1].reshape(n, h, -1, 1)
        
        d = keys.size()[-1]
        keys = keys.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        values = values.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        mask = mask.expand(-1, h, -1, -1).transpose(-1, -2).gather(dim=-2, index=sparse_idx).transpose(-1, -2)

        return keys, values, mask

    def get_sparse_tokens_with_pooling(self, keys, values, mask):
        
        if self.sparsity_factor == 1:
            return keys, values, mask.expand(-1, keys.size()[1], -1, -1)

        keys = self.chunk(keys, self.sparsity_factor)
        values = self.chunk(values, self.sparsity_factor)

        n, h, b, t, d = keys.size()
        mask = mask.reshape(n, 1, b, 1, t)
        mask = ~mask.transpose(-1, -2).bool()

        keys = keys * mask
        values = values * mask

        mask = mask.sum(dim=-2)
        keys = keys.sum(dim=-2) / (mask + 1e-6)
        values = values.sum(dim=-2) / (mask + 1e-6)

        mask = - (1. - mask.clamp(0, 1)) * 1e4
        return keys.reshape(n, h, -1, d), values.reshape(n, h, -1, d), mask.expand(-1, h, -1, -1).transpose(-1, -2)

    def get_sparse_tokens_with_stride(self, keys, values, mask):

        if self.sparsity_factor == 1:
            return keys, values, mask.expand(-1, keys.size()[1], -1, -1)

        n, h, t, d = keys.size()
        sparse_idx = torch.arange(t // self.sparsity_factor, device=keys.device) * self.sparsity_factor
        sparse_idx = sparse_idx.reshape(1, 1, -1, 1) + (torch.arange(h, device=keys.device) % self.sparsity_factor).reshape(1, h, 1, 1)
        sparse_idx = sparse_idx.expand(n, h, -1, 1)

        keys = keys.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        values = values.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        mask = mask.expand(-1, h, -1, -1).transpose(-1, -2).gather(dim=-2, index=sparse_idx).transpose(-1, -2)

        return keys, values, mask

    def get_sparse_tokens_with_block_stride(self, keys, values, mask):

        if self.sparsity_factor == 1:
            return keys, values, mask.expand(-1, keys.size()[1], -1, -1)

        n, h, t, d = keys.size()

        t, b = self.block_size, t // self.block_size
        sparse_idx = torch.arange(t // self.sparsity_factor, device=keys.device)
        sparse_idx = sparse_idx.reshape(1, 1, 1, -1, 1) + torch.arange(h, device=keys.device).reshape(1, h, 1, 1, 1) * (t // self.sparsity_factor)
        sparse_idx = (sparse_idx % t) 
        sparse_idx = sparse_idx + torch.arange(b, device=keys.device).reshape(1, 1, -1, 1, 1) * t
        sparse_idx = sparse_idx.reshape(1, h, -1, 1).expand(n, h, -1, 1)

        keys = keys.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        values = values.gather(dim=-2, index=sparse_idx.expand(-1, -1, -1, d))
        mask = mask.expand(-1, h, -1, -1).transpose(-1, -2).gather(dim=-2, index=sparse_idx).transpose(-1, -2)

        return keys, values, mask
        
    def get_sparse_tokens_with_lsh(self, keys, values, mask):
        
        if self.sparsity_factor == 1:
            return keys, values, mask.expand(-1, keys.size()[1], -1, -1)

        block_size = min(self.block_size, self.sparse_block_size)
        keys = self.chunk(keys, block_size)
        values = self.chunk(values, block_size)

        n, h, b, t, d = keys.size()
        mask = mask.reshape(n, 1, b, 1, t)
        mask = ~mask.transpose(-1, -2).bool()

        keys = keys * mask
        values = values * mask
        mask = mask.expand(-1, h, -1, -1, -1).float()

        extra_factor = 1
        
        for _ in range(self.lsh_num_pre_rounds):
            keys, values, mask = self.lsh_round(keys, values, mask, t*extra_factor)

        keys, values, mask = self.lsh_round(keys, values, mask, t//self.sparsity_factor)
        keys /= mask + 1e-8
        values /= mask + 1e-8

        mask = -10000 * (1. - mask.clamp(0, 1))

        return keys.reshape(n, h, -1, d), values.reshape(n, h, -1, d), mask.transpose(-1, -2).reshape(n, h, 1, -1)

    def lsh_round(self, keys, values, mask, output_size):

        with torch.no_grad():

            n_hashes = output_size // 2
            n, h, b, t, d = keys.size()
            binary_mask = mask.clamp(0, 1)

            indexes = (torch.nn.functional.normalize(keys, dim=-1) * binary_mask) @ torch.randn(1, h, 1, d, n_hashes, device=keys.device)
            indexes = torch.cat([indexes, -indexes], dim=-1).argmax(dim=-1, keepdim=True)

        n, h, b, t, d = keys.size()
        
        x_ = torch.zeros(n, h, b, output_size, d, device=keys.device)
        mask_ = torch.zeros(n, h, b, output_size, 1, device=keys.device)
        keys = torch.scatter_add(x_, dim=-2, index=indexes.expand(-1, -1, -1, -1, d), src=keys)
        values = torch.scatter_add(x_, dim=-2, index=indexes.expand(-1, -1, -1, -1, d), src=values)
        mask = torch.scatter_add(mask_, dim=-2, index=indexes, src=mask)

        return keys[..., :output_size, :], values[..., :output_size, :], mask[..., :output_size, :]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):
        
        query_layer = self.q_lin(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.k_lin(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.v_lin(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.k_lin(hidden_states))
            value_layer = self.transpose_for_scores(self.v_lin(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.k_lin(hidden_states))
            value_layer = self.transpose_for_scores(self.v_lin(hidden_states))

        query_layer = self.transpose_for_scores(query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

            if is_cross_attention:
                outputs = self.cross_attention_forward(
                    query_layer=query_layer, 
                    key_layer=key_layer, 
                    value_layer=value_layer, 
                    attention_mask=attention_mask,
                    output_attentions=output_attentions
                    )
            else:
                outputs = self.causal_forward(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            outputs = outputs + ((key_layer, value_layer),)
            
        else:
            outputs = self.not_causal_forward(
                query_layer,
                key_layer,
                value_layer, 
                attention_mask=attention_mask, 
                output_attentions=output_attentions
                )
        
        #if head_mask is not None:
        #    outputs = (outputs[0] * head_mask[:, :, :1, :1], ) + outputs[1:]
        return (self.out_lin(outputs[0]),) + outputs[1:]

    def causal_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        output_attentions=False,
        ):

        n, h, t, d = key_layer.size()

        # Cat global mask
        attention_mask = torch.nn.functional.pad(attention_mask, (self.num_global_tokens, 0), value=0)

        # Split input into global tokens and other tokens
        split = (self.num_global_tokens, t - self.num_global_tokens)
        global_query, query_layer = query_layer.split(split, dim=-2)

        # Use normal causal attention if local attention covers every tokens
        if t <= 2 * self.block_size + self.num_global_tokens:
            context_layer = self.causal_attention(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask,
                causal_shape=(t - self.num_global_tokens, t - self.num_global_tokens)
                )
            
            context_layer = torch.cat([global_query, context_layer], dim=-2)
            return (self.reshape_output(context_layer), )
        
        # Split K Q M on global and non global
        global_key, key_layer = key_layer.split(split, dim=-2)
        global_value, value_layer = value_layer.split(split, dim=-2)
        global_mask, attention_mask = attention_mask.split(split, dim=-1)
        
        n, h, t, d = key_layer.size()

        # Get sparse idx
        sparse_key, sparse_value, sparse_mask = (None, None, None)
        if self.sparse_block_size and self.sparsity_factor > 0:
            sparse_key, sparse_value, sparse_mask = self.get_sparse_elements(key_layer, value_layer, attention_mask)
        
        # Expand masks on heads
        attention_mask = attention_mask.expand(-1, h, -1, -1)
        global_mask = global_mask.expand(-1, h, -1, -1)

        # Compute dot product attention
        context_layer = self.attention(
            query_layer, 
            key_layer, 
            value_layer, 
            attention_mask,
            sparse_key=sparse_key,
            sparse_value=sparse_value,
            sparse_mask=sparse_mask,
            global_key=global_key,
            global_value=global_value,
            global_mask=global_mask
            )

        # Merge pseudo global (causal) and local-sparse tokens
        context_layer = torch.cat([global_query, context_layer], dim=-2)
        context_layer = self.reshape_output(context_layer)

        return (context_layer,)

    def not_causal_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        output_attentions=False,
        ):

        n, h, t, d = query_layer.size()

        # Cat global mask
        attention_mask = torch.nn.functional.pad(attention_mask, (self.num_global_tokens, 0), value=0)
        
        # Use normal attention if local attention covers every tokens
        if t <= 2 * self.block_size + self.num_global_tokens:
            context_layer = self.full_attention(
                query_layer=query_layer, 
                key_layer=key_layer, 
                value_layer=value_layer, 
                attention_mask=attention_mask
                )
            return (self.reshape_output(context_layer), )

        # Split input into global tokens and other tokens
        split = (self.num_global_tokens, t - self.num_global_tokens)
        global_query, query_layer = query_layer.split(split, dim=-2)
        
        # Get global_attention
        bos = self.full_attention(
            query_layer=global_query, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask
            )
        
        # Split K Q M on global and non global
        global_key, key_layer = key_layer.split(split, dim=-2)
        global_value, value_layer = value_layer.split(split, dim=-2)
        global_mask, attention_mask = attention_mask.split(split, dim=-1)
        
        n, h, t, d = key_layer.size()

        # Get sparse idx
        sparse_key, sparse_value, sparse_mask = (None, None, None)

        if self.sparse_block_size and self.sparsity_factor > 0:
            sparse_key, sparse_value, sparse_mask = self.get_sparse_elements(key_layer, value_layer, attention_mask)
        
        # Expand masks on heads
        attention_mask = attention_mask.expand(-1, h, -1, -1)
        global_mask = global_mask.expand(-1, h, -1, -1)

        # Compute dot product attention
        context_layer = self.attention(
            query_layer, 
            key_layer, 
            value_layer, 
            attention_mask,
            sparse_key=sparse_key, 
            sparse_value=sparse_value, 
            sparse_mask=sparse_mask,
            global_key=global_key,
            global_value=global_value,
            global_mask=global_mask
            )

        # Merge global and local-sparse tokens
        context_layer = torch.cat([bos, context_layer], dim=-2)
        context_layer = self.reshape_output(context_layer)
        
        return (context_layer,)

    def cross_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        output_attentions=False,
        ):

        context_layer = self.full_attention(
            query_layer=query_layer, 
            key_layer=key_layer, 
            value_layer=value_layer, 
            attention_mask=attention_mask
        )
        return (self.reshape_output(context_layer), )

    def chunk(self, x, chunk_size):

        n, h, t, d = x.size()
        return x.reshape(n, h, -1, chunk_size, d)


class LSGTransformerBlock(nn.Module):

    def __init__(self, config):

        nn.Module.__init__(self)

        assert config.dim % config.n_heads == 0

        self.attention = LSGSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            hidden_states=x,
            attention_mask=-10000*(1 - attn_mask).unsqueeze(1).unsqueeze(1),
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class LSGTransformer(Transformer):

    def __init__(self, config):

        nn.Module.__init__(self)

        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([LSGTransformerBlock(config) for _ in range(config.n_layers)])


class LSGDistilBertPreTrainedModel(DistilBertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LSGDistilBertConfig


class LSGDistilBertModel(LSGDistilBertPreTrainedModel, DistilBertModel):

    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.embeddings = LSGEmbeddings(config)  # Embeddings
        self.transformer = LSGTransformer(config)  # Encoder

        assert hasattr(config, "num_global_tokens")
        self.num_global_tokens = config.num_global_tokens
        self.pad_idx = config.pad_token_id

        assert hasattr(config, "block_size") and hasattr(config, "adaptive")
        self.block_size = config.block_size
        self.adaptive = config.adaptive
        self.mask_first_token = config.first_token
        self.pool_with_global = config.pool_with_global

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        inputs_ = input_ids if input_ids is not None else inputs_embeds
        n, t = inputs_.size()[:2]

        if attention_mask is None:
            attention_mask = torch.ones(n, t, device=inputs_.device)
        if self.mask_first_token:
            attention_mask[:,0] = 0
            
        b = self.block_size * 2
        pad = t % self.block_size
        
        # Check if t is multiple of block_size and pad
        if self.adaptive and t > b and pad > 0:
            pad_length = self.block_size - pad
            if input_ids is not None:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=self.pad_idx)
            else:
                inputs_embeds = torch.nn.functional.pad(inputs_embeds.transpose(-1, -2), (0, pad_length), value=0.).transpose(-1, -2)
            
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=0)
        
        n, t_ = attention_mask.size()

        encoder_outputs = self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )

        context = encoder_outputs[0]
        if self.pool_with_global:
            context[:, self.num_global_tokens] = context[:, 0]
        
        diff = t - t_
        n, _, d = context.size()
        context = context[..., self.num_global_tokens:, :]

        # Adapt sequence to initial shape
        if diff < 0:
            context = context[:, :t]
        
        if not return_dict:
            return (context, ) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=context, 
            hidden_states=encoder_outputs.hidden_states, 
            attentions=encoder_outputs.attentions,
        )

    def _forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        inputs_embeds = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LSGDistilBertForMaskedLM(LSGDistilBertPreTrainedModel, DistilBertForMaskedLM):

    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.activation = get_activation(config.activation)
        
        self.distilbert = LSGDistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()


class LSGDistilBertForSequenceClassification(LSGDistilBertPreTrainedModel, DistilBertForSequenceClassification):

    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = LSGDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()


class LSGDistilBertForQuestionAnswering(LSGDistilBertPreTrainedModel, DistilBertForQuestionAnswering):

    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.distilbert = LSGDistilBertModel(config)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        # Initialize weights and apply final processing
        self.post_init()


class LSGDistilBertForTokenClassification(LSGDistilBertPreTrainedModel, DistilBertForTokenClassification):

    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.num_labels = config.num_labels

        self.distilbert = LSGDistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


class LSGDistilBertForMultipleChoice(LSGDistilBertPreTrainedModel, DistilBertForMultipleChoice):
    
    def __init__(self, config):

        LSGDistilBertPreTrainedModel.__init__(self, config)

        self.distilbert = LSGDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Register model in Auto API
try:
    LSGDistilBertConfig.register_for_auto_class()
    for key, value in AUTO_MAP.items():
        str_to_class(value.split(".")[-1]).register_for_auto_class(key)
except:
    warn("AutoRegister isn't available, you'll have to manually copy modeling.py after .save_pretrained(...).")
    warn("Update to transformers >= 4.17.0 to fix.")
