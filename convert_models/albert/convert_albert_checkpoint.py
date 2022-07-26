from .modeling_lsg_albert import *
from ..conversion_utils import ConversionScript


class AlbertConversionScript(ConversionScript):

    _ARCHITECTURE_TYPE_DICT = {
        "AlbertModel": ("LSGAlbertModel", LSGAlbertModel),
        "AlbertForMaskedLM": ("LSGAlbertForMaskedLM", LSGAlbertForMaskedLM),
        "AlbertForPreTraining": ("LSGAlbertForPreTraining", LSGAlbertForPreTraining),
        "AlbertForMaskedLM": ("LSGAlbertForMaskedLM", LSGAlbertForMaskedLM),
        "AlbertForMultipleChoice": ("LSGAlbertForMultipleChoice", LSGAlbertForMultipleChoice),
        "AlbertForQuestionAnswering": ("LSGAlbertForQuestionAnswering", LSGAlbertForQuestionAnswering),
        "AlbertForSequenceClassification": ("LSGAlbertForSequenceClassification", LSGAlbertForSequenceClassification),
        "AlbertForTokenClassification": ("LSGAlbertForTokenClassification", LSGAlbertForTokenClassification)
    }

    _ARCHITECTURE_TYPE_DICT = {**{"LSG" + k: v for k, v in _ARCHITECTURE_TYPE_DICT.items()}, **_ARCHITECTURE_TYPE_DICT}
    _BASE_ARCHITECTURE_TYPE = "AlbertModel"
    _DEFAULT_ARCHITECTURE_TYPE = "AlbertForPreTraining"
    _CONFIG_MODULE = LSGAlbertConfig

    _DEFAULT_CONFIG_POSITIONAL_OFFSET = 0
    _DEFAULT_POSITIONAL_OFFSET = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_module(self, model, is_base_architecture):
        if is_base_architecture:
            return model
        return model.albert

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):

        import torch
        from torch.distributions.multivariate_normal import MultivariateNormal

        u = module_prefix.embeddings.word_embeddings.weight.clone()
        cov = torch.cov(u.T)
        m = MultivariateNormal(u.mean(dim=0), cov)
        w = m.sample((512,))
        w[0] = u[bos_id]

        positions = module_prefix.embeddings.position_embeddings.weight.clone()
        positions = self.order_positions(positions, stride)

        if self.use_token_ids:
            token_ids = module_prefix.embeddings.token_type_embeddings.weight.clone()
            positions += token_ids[0].unsqueeze(0)
            w[0] = u[bos_id] + token_ids[0]

        if keep_first_global:
            module_prefix.embeddings.global_embeddings.weight.data[1:] = (w + positions)[1:]
        else:
            module_prefix.embeddings.global_embeddings.weight.data = w + positions

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):

        u = module_prefix.embeddings.word_embeddings.weight.clone()
        positions = module_prefix.embeddings.position_embeddings.weight.clone()
        positions = self.order_positions(positions, stride)

        positions[0] += u[bos_id]
        positions[1:] += u[mask_id].unsqueeze(0)

        if self.use_token_ids:
            token_ids = module_prefix.embeddings.token_type_embeddings.weight.clone()
            positions += token_ids[0].unsqueeze(0)
        
        if keep_first_global:
            module_prefix.embeddings.global_embeddings.weight.data[1:] = positions[1:]
        else:
            module_prefix.embeddings.global_embeddings.weight.data = positions
        
    def update_positions(self, module_prefix, max_pos):

        position_embeddings_weights = module_prefix.embeddings.position_embeddings.weight.clone()
        current_max_position = position_embeddings_weights.size()[0]

        new_position_embeddings_weights = torch.cat([
            position_embeddings_weights for _ in range(max_pos//current_max_position + 1)
            ], dim=0)[:max_pos]

        module_prefix.embeddings.position_ids = torch.arange(max_pos, device=module_prefix.embeddings.position_ids.device).unsqueeze(0)
        module_prefix.embeddings.position_embeddings.weight.data = new_position_embeddings_weights