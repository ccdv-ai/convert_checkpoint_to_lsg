from logging import warn
import torch
from transformers.models.mbart.modeling_mbart import *
from transformers.models.mbart.modeling_mbart import _expand_mask
import torch.nn as nn
import sys

AUTO_MAP = {
        "AutoModel": "modeling_lsg_mbart.LSGMBartModel",
        "AutoModelForCausalLM": "modeling_lsg_mbart.LSGMBartForCausalLM",
        "AutoModelForQuestionAnswering": "modeling_lsg_mbart.LSGMBartForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_lsg_mbart.LSGMBartForSequenceClassification",
        "AutoModelForSeq2SeqLM": "modeling_lsg_mbart.LSGMBartForConditionalGeneration"
    }

class LSGMBartConfig(MBartConfig):
    """
    This class overrides :class:`~transformers.MBartConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    base_model_prefix = "lsg"
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        adaptive=True,
        base_model_prefix="lsg",
        block_size=128,
        lsh_num_pre_rounds=1,
        mask_first_token=False,
        num_global_tokens=1,
        pass_global_tokens_to_decoder=True,
        pool_with_global=True,
        sparse_block_size=128,
        sparsity_factor=2,
        sparsity_type="norm",
        **kwargs
        ):
        """Constructs LSGConfig."""
        super().__init__(**kwargs)

        self.adaptive = adaptive
        self.auto_map = AUTO_MAP
        self.base_model_prefix = base_model_prefix
        self.block_size = block_size
        self.lsh_num_pre_rounds = lsh_num_pre_rounds
        self.mask_first_token = mask_first_token
        self.num_global_tokens = num_global_tokens
        self.pass_global_tokens_to_decoder = pass_global_tokens_to_decoder
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

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        is_decoder=False,
        bias=True,
        ):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.head_dim,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_output(self, context_layer):
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        return context_layer.view(*new_context_layer_shape)

    def project_QKV(self, hidden_states):

        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))
        key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
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


class LSGAttentionProduct(nn.Module):

    def __init__(self, config, block_size=None, sparse_block_size=None, sparsity_factor=4):
        """
        Compute block or overlapping blocks attention products
        """
        super().__init__()
 
        self.block_size = block_size
        self.sparse_block_size = sparse_block_size
        self.sparsity_factor = sparsity_factor

        if self.block_size is None:
            self.block_size = config.block_size

        if self.sparse_block_size is None:
            self.sparse_block_size = config.sparse_block_size

        # Shape of blocks
        self.local_shapes = (self.block_size*3, self.block_size)
        if self.sparse_block_size and self.sparsity_factor > 0:
            self.sparse_shapes = (self.sparse_block_size*3, self.block_size//self.sparsity_factor)

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


class LSGMBartEncoderAttention(BaseSelfAttention):
    '''
    Compute local attention with overlapping blocs
    Use global attention for tokens with highest norm
    '''
    def __init__(
        self, 
        config, 
        embed_dim,
        num_heads,
        dropout
        ):

        super().__init__(embed_dim, num_heads, dropout)

        self.block_size = config.block_size
        self.sparse_block_size = config.sparse_block_size
        self.num_global_tokens = config.num_global_tokens
        self.sparsity_factor = config.sparsity_factor

        self.attention = LSGAttentionProduct(
            config, 
            block_size=config.block_size, 
            sparse_block_size=config.sparse_block_size, 
            sparsity_factor=self.sparsity_factor, 
            )

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
        layer_head_mask=None,
        output_attentions=False
        ):

        query_layer, key_layer, value_layer = self.project_QKV(hidden_states)
        outputs = self.not_causal_forward(
            query_layer,
            key_layer,
            value_layer, 
            attention_mask=attention_mask[:, :, :1, :], 
            head_mask=layer_head_mask, 
            output_attentions=output_attentions
            )
        
        return self.out_proj(outputs), None, None

    def not_causal_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        head_mask=None,
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

            return self.reshape_output(context_layer)

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
        
        return context_layer

    def chunk(self, x, chunk_size):

        n, h, t, d = x.size()
        return x.reshape(n, h, -1, chunk_size, d)


class LSGMBartEncoderLayer(MBartEncoderLayer):

    def __init__(self, config):

        super().__init__(config)
        self.self_attn = LSGMBartEncoderAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )


class LSGMBartPretrainedModel(MBartPreTrainedModel):

    config_class = LSGMBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MBartDecoder, MBartEncoder, LSGMBartEncoder)):
            module.gradient_checkpointing = value


class LSGMBartEncoder(LSGMBartPretrainedModel, MBartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MBartEncoderLayer`].
    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config, embed_tokens=None):

        LSGMBartPretrainedModel.__init__(self, config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([LSGMBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 
        assert hasattr(config, "num_global_tokens")
        self.num_global_tokens = config.num_global_tokens
        self.pad_idx = config.pad_token_id

        assert hasattr(config, "block_size") and hasattr(config, "adaptive")
        self.block_size = config.block_size
        self.adaptive = config.adaptive
        self.mask_first_token = config.mask_first_token
        self.pool_with_global = config.pool_with_global
        self.pass_global_tokens_to_decoder = config.pass_global_tokens_to_decoder

        self.global_embeddings = nn.Embedding(512, embedding_dim=config.d_model)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
        ):

        
        inputs_ = input_ids if input_ids is not None else inputs_embeds
        n, t = inputs_.size()[:2]

        if attention_mask is None:
            attention_mask = torch.ones(n, t, device=inputs_.device, dtype=inputs_.dtype)
        if self.mask_first_token:
            attention_mask[:, 0] = 0

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
        
        encoder_outputs = self.forward_with_adaptive(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        
        context = encoder_outputs[0]
        diff = t - t_

        if self.pass_global_tokens_to_decoder:
            offset = self.num_global_tokens
        else:
            if self.pool_with_global:
                context[:, self.num_global_tokens] = context[:, 0]
            context = context[..., self.num_global_tokens:, :]
            offset = 0

        # Adapt sequence to initial shape
        if diff < 0:
            context = context[:, :t + offset]
        
        if return_dict:
            encoder_outputs.last_hidden_state = context
        else:
            encoder_outputs = (context, ) + encoder_outputs[1:]
        
        return encoder_outputs

    def forward_with_adaptive(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(inputs_embeds)
        hidden_states = inputs_embeds + embed_pos

        # Add global tokens
        n, t, d = hidden_states.size()
        global_idx = torch.arange(self.num_global_tokens, device=hidden_states.device).reshape(1, -1)
        hidden_states = torch.cat([self.global_embeddings(global_idx).expand(n, -1, -1), hidden_states], dim=-2)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class LSGMBartModel(LSGMBartPretrainedModel, MBartModel):

    def __init__(self, config):

        LSGMBartPretrainedModel.__init__(self, config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.pass_global_tokens_to_decoder = config.pass_global_tokens_to_decoder
        self.num_global_tokens = config.num_global_tokens

        self.encoder = LSGMBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Pad mask for global tokens
        if self.pass_global_tokens_to_decoder and attention_mask is not None:
            attention_mask = torch.nn.functional.pad(attention_mask, pad=(self.num_global_tokens, 0), value=1)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class LSGMBartForConditionalGeneration(LSGMBartPretrainedModel, MBartForConditionalGeneration):
    
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
    ]

    def __init__(self, config):

        LSGMBartPretrainedModel.__init__(self, config)
        self.model = LSGMBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
    
        # Initialize weights and apply final processing
        self.post_init()


class LSGMBartForSequenceClassification(LSGMBartPretrainedModel, MBartForSequenceClassification):

    def __init__(self, config, **kwargs):

        LSGMBartPretrainedModel.__init__(self, config, **kwargs)
        self.model = LSGMBartModel(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)


class LSGMBartForQuestionAnswering(LSGMBartPretrainedModel, MBartForQuestionAnswering):

    def __init__(self, config):

        LSGMBartPretrainedModel.__init__(self, config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = LSGMBartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)


class LSGMBartForCausalLM(LSGMBartPretrainedModel, MBartForCausalLM):

    def __init__(self, config):

        LSGMBartPretrainedModel.__init__(self, config)
        MBartForCausalLM.__init__(self, config)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Register model in Auto API
try:
    LSGMBartConfig.register_for_auto_class()
    for key, value in AUTO_MAP.items():
        str_to_class(value.split(".")[-1]).register_for_auto_class(key)
except:
    warn("AutoRegister isn't available, you'll have to manually copy modeling.py after .save_pretrained(...).")
    warn("Update to transformers >= 4.23.1 to fix.")