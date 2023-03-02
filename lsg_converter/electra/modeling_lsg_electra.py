from logging import warn
from transformers.models.electra.modeling_electra import *
import torch
import torch.nn as nn
from transformers.models.electra.configuration_electra import ElectraConfig
import sys

AUTO_MAP = {
        "AutoModel": "modeling_lsg_electra.LSGElectraModel",
        "AutoModelForCausalLM": "modeling_lsg_electra.LSGElectraForCausalLM",
        "AutoModelForMaskedLM": "modeling_lsg_electra.LSGElectraForMaskedLM",
        "AutoModelForPreTraining": "modeling_lsg_electra.LSGElectraForPreTraining",
        "AutoModelForMultipleChoice": "modeling_lsg_electra.LSGElectraForMultipleChoice",
        "AutoModelForQuestionAnswering": "modeling_lsg_electra.LSGElectraForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_lsg_electra.LSGElectraForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_lsg_electra.LSGElectraForTokenClassification"
    }

class LSGElectraConfig(ElectraConfig):
    """
    This class overrides :class:`~transformers.ElectraConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    base_model_prefix = "lsg"
    model_type = "electra"

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
        """Constructs LSGElectraConfig."""
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
                "[WARNING CONFIG]: sparsity_mode not in [None, 'none', 'norm', 'lsh', 'pooling', 'stride', 'block_stride'], \
                    setting sparsity_type=None, computation will skip sparse attention")
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
                "[WARNING CONFIG]: num_global_tokens > 512 is not allowed, setting num_global_tokens=512"
            )
            self.num_global_tokens = 512
        
        if self.sparsity_factor > 0:
            assert self.block_size % self.sparsity_factor == 0, "[ERROR CONFIG]: block_size must be divisible by sparsity_factor"
            assert self.block_size//self.sparsity_factor >= 1, "[ERROR CONFIG]: make sure block_size >= sparsity_factor"
            
        if self.mask_first_token and not pool_with_global:
            logger.warning(
                "[WARNING CONFIG]: pool_with_global==False is not compatible with mask_first_token==True. Setting pool_with_global to True.")
            self.pool_with_global = True
        
        if hasattr(self, "position_embedding_type"):
            if self.position_embedding_type != "absolute":
                logger.warning(
                "[WARNING CONFIG]: LSG Attention is not compatible with relative positional embedding and will skip its computation. Set position_embedding_type='absolute' to remove this warning.")
        
        
class BaseSelfAttention(nn.Module):
    
    def init_modules(self, config):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_output(self, context_layer):
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_layer_shape)

    def project_QKV(self, hidden_states):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        return query_layer, key_layer, value_layer


class BaseAttentionProduct(nn.Module):

    def __init__(self, config):
        """
        Compute attention: softmax(Q @ K.T) @ V
        """
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
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
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.block_size = config.block_size

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None, causal_shape=None):
        
        d = query_layer.shape[-1]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2) / math.sqrt(d)

        del query_layer
        del key_layer

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
            attention_scores = attention_scores + attention_mask

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

        # Input batch, heads, length, hidden_size
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

    def reshape_to_sparse_block(self, hidden_states, is_attn_mask=False):
        
        size, step = self.sparse_shapes

        # In case of odd case
        odd_offset = (step % 2)

        # n, h, t, d*2 + 1
        size = size*2 
        s = (size - step) // 2 + odd_offset

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


class LSGElectraEmbeddings(ElectraEmbeddings):

    def __init__(self, config):
        super().__init__(config)

        self.num_global_tokens = config.num_global_tokens

        # Hardcoded but partially trained
        self.global_embeddings = nn.Embedding(512, embedding_dim=config.embedding_size, )

        self.block_size = config.block_size

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids[:, :seq_length])

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids[:, :seq_length])
            embeddings += position_embeddings

        #if self.num_global_tokens < 0:
        n, t, d = embeddings.size()
        
        # Add global_tokens
        indexes = torch.arange(self.num_global_tokens, device=embeddings.device).reshape(1, -1)
        global_embeddings = self.global_embeddings(indexes) 
        embeddings = torch.cat([global_embeddings.expand(n, -1, d), embeddings], dim=-2)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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

        mask = (1. - mask.clamp(0, 1)) 
        mask *= torch.finfo(mask.dtype).min
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

        mask = (1. - mask.clamp(0, 1)) 
        mask *= torch.finfo(mask.dtype).min

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

        query_layer = self.query(hidden_states)

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
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

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
        
        return outputs

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


class LSGAttention(ElectraAttention):

    def __init__(self, config):

        nn.Module.__init__(self)
        
        self.self = LSGSelfAttention(config)
        self.output = ElectraSelfOutput(config)
        self.pruned_heads = set()


class LSGElectraLayer(ElectraLayer):
    
    def __init__(self, config):

        super().__init__(config)

        self.attention = LSGAttention(config)
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = LSGAttention(config, position_embedding_type="absolute")


class LSGElectraEncoder(ElectraEncoder):

    def __init__(self, config):

        super().__init__(config)

        self.layer = nn.ModuleList([LSGElectraLayer(config) for _ in range(config.num_hidden_layers)])

        assert hasattr(config, "num_global_tokens")
        self.num_global_tokens = config.num_global_tokens
        self.pad_idx = config.pad_token_id

        assert hasattr(config, "block_size") and hasattr(config, "adaptive")
        self.block_size = config.block_size
        self.adaptive = config.adaptive
        self.mask_first_token = config.mask_first_token
        self.pool_with_global = config.pool_with_global

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        mask_value = torch.finfo(attention_mask.dtype).min
        n, _, __, t = attention_mask.size()
        
        if not (self.config.is_decoder and encoder_hidden_states is not None):
            b = self.block_size * 2
            pad = t % self.block_size
            
            # Check if t is multiple of block_size and pad
            if self.adaptive and t > b and pad > 0:
                pad_length = self.block_size - pad
                hidden_states = torch.nn.functional.pad(hidden_states.transpose(-1, -2), (0, pad_length), value=0.).transpose(-1, -2)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=mask_value)

            if self.mask_first_token:
                attention_mask[..., 0] = mask_value

        encoder_outputs = super().forward(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )

        sequence_output = encoder_outputs[0]
        if self.pool_with_global:
            sequence_output[:, self.num_global_tokens] = sequence_output[:, 0]

        # Adapt sequence to initial shape
        sequence_output = sequence_output[..., self.num_global_tokens: t + self.num_global_tokens, :]

        if not return_dict:
            return (sequence_output, ) + encoder_outputs[1:]
        
        encoder_outputs.last_hidden_state = sequence_output 
        return encoder_outputs

class LSGElectraPreTrainedModel(ElectraPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LSGElectraConfig
    load_tf_weights = load_tf_weights_in_electra
    base_model_prefix = "electra"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"electra.embeddings_project.weight", r"electra.embeddings_project.bias"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ElectraEncoder, LSGElectraEncoder)):
            module.gradient_checkpointing = value


class LSGElectraModel(LSGElectraPreTrainedModel, ElectraModel):
    """
    This class overrides :class:`~transformers.ElectraModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = LSGElectraConfig

    def __init__(self, config):
        
        LSGElectraPreTrainedModel.__init__(self, config)

        self.embeddings = LSGElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = LSGElectraEncoder(config)
        self.config = config
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None):

        # Do not rely on original triangular mask from BERT/RoBERTa for causalLM
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min

        return extended_attention_mask


class LSGElectraForPreTraining(LSGElectraPreTrainedModel, ElectraForPreTraining):

    config_class = LSGElectraConfig

    def __init__(self, config):

        LSGElectraPreTrainedModel.__init__(self, config)

        self.electra = LSGElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        # Initialize weights and apply final processing
        self.post_init()


class LSGElectraForMaskedLM(LSGElectraPreTrainedModel, ElectraForMaskedLM):
    """
    This class overrides :class:`~transformers.ElectraForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = LSGElectraConfig

    def __init__(self, config):

        LSGElectraPreTrainedModel.__init__(self, config)

        self.electra = LSGElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)

        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # Initialize weights and apply final processing
        self.post_init()


class LSGElectraForSequenceClassification(LSGElectraPreTrainedModel, ElectraForSequenceClassification):
    """
    This class overrides :class:`~transformers.ElectraForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = LSGElectraConfig

    def __init__(self, config):
        
        LSGElectraPreTrainedModel.__init__(self, config)

        self.num_labels = config.num_labels
        self.config = config
        self.electra = LSGElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        
class LSGElectraForMultipleChoice(LSGElectraPreTrainedModel, ElectraForMultipleChoice):
    """
    This class overrides :class:`~transformers.ElectraForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = LSGElectraConfig

    def __init__(self, config):
        
        LSGElectraPreTrainedModel.__init__(self, config)

        self.electra = LSGElectraModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()


class LSGElectraForCausalLM(LSGElectraPreTrainedModel, ElectraForCausalLM):

    def __init__(self, config):

        LSGElectraPreTrainedModel.__init__(self, config)

        if not config.is_decoder:
            logger.warning("If you want to use `ElectraForCausalLM` as a standalone, add `is_decoder=True.`")

        self.electra = LSGElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        self.init_weights()


class LSGElectraForTokenClassification(LSGElectraPreTrainedModel, ElectraForTokenClassification):
    """
    This class overrides :class:`~transformers.ElectraForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = LSGElectraConfig

    def __init__(self, config):
        
        LSGElectraPreTrainedModel.__init__(self, config)

        self.num_labels = config.num_labels

        self.electra = LSGElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


class LSGElectraForQuestionAnswering(LSGElectraPreTrainedModel, ElectraForQuestionAnswering):
    """
    This class overrides :class:`~transformers.ElectraForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = LSGElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config):
        
        LSGElectraPreTrainedModel.__init__(self, config)

        self.num_labels = config.num_labels

        self.electra = LSGElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Register model in Auto API
try:
    LSGElectraConfig.register_for_auto_class()
    for key, value in AUTO_MAP.items():
        str_to_class(value.split(".")[-1]).register_for_auto_class(key)
except:
    warn("AutoRegister isn't available, you'll have to manually copy modeling.py after .save_pretrained(...).")
    warn("Update to transformers >= 4.23.1 to fix.")