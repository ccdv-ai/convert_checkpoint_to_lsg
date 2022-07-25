import os
import sys
from dataclasses import dataclass, field

from pegasus.modeling_lsg_pegasus import *
import warnings
import json 

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from .conversion_utils import ConversionScript

class PegasusConversionScript(ConversionScript):

    _ARCHITECTURE_TYPE_DICT = {
        "PegasusModel": ("LSGPegasusModel", LSGPegasusModel),
        "PegasusForCausalLM": ("LSGPegasusForCausalLM", LSGPegasusForCausalLM),
        "PegasusForConditionalGeneration": ("LSGPegasusForConditionalGeneration", LSGPegasusForConditionalGeneration),
    }
    _ARCHITECTURE_TYPE_DICT = {**{"LSG" + k: v for k, v in _ARCHITECTURE_TYPE_DICT.items()}, **_ARCHITECTURE_TYPE_DICT}

    _BASE_ARCHITECTURE_TYPE = "PegasusModel"
    _DEFAULT_ARCHITECTURE_TYPE = "PegasusForConditionalGeneration"
    _CONFIG_MODULE = LSGPegasusConfig

    _DEFAULT_CONFIG_POSITIONAL_OFFSET = 0
    _DEFAULT_POSITIONAL_OFFSET = 0

    def __init__(
        self, 
        initial_model, 
        model_name, 
        max_sequence_length, 
        architecture, 
        random_global_init, 
        global_positional_stride, 
        keep_first_global_token, 
        resize_lsg, 
        model_kwargs, 
        config,
        seed
        ):
        super().__init__(
            initial_model, 
            model_name, 
            max_sequence_length, 
            architecture, 
            random_global_init, 
            global_positional_stride, 
            keep_first_global_token, 
            resize_lsg, 
            model_kwargs, 
            config,
            seed
        )

    def get_module(self, model, is_base_architecture):
        if is_base_architecture:
            return model
        return model.model

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):

        import torch
        from torch.distributions.multivariate_normal import MultivariateNormal

        u = module_prefix.shared.weight.clone()
        cov = torch.cov(u.T)
        m = MultivariateNormal(u.mean(dim=0), cov)
        w = m.sample((512,))
        
        w[0] = u[bos_id]
        positions = module_prefix.encoder.embed_positions.weight.clone()
        positions = self.order_positions(positions, stride)

        if keep_first_global:
            module_prefix.encoder.global_embeddings.weight.data[1:] = (w + positions)[1:]
        else:
            module_prefix.encoder.global_embeddings.weight.data = w + positions

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):

        u = module_prefix.shared.weight.clone()
        positions = module_prefix.encoder.embed_positions.weight.clone()
        positions = self.order_positions(positions, stride)
        
        positions[0] += u[bos_id]
        positions[1:] += u[mask_id].unsqueeze(0)

        if keep_first_global:
            module_prefix.encoder.global_embeddings.weight.data[1:] = positions[1:]
        else:
            module_prefix.encoder.global_embeddings.weight.data = positions

    def update_positions_with_model(self, model, max_pos):
        model.resize_position_embeddings(max_pos)