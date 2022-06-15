import os
import sys
from dataclasses import dataclass, field

from sklearn import model_selection

from barthez.modeling_lsg_barthez import *
import warnings

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)


_MODEL_TYPE_DICT = {
    "MBartModel": ("LSGMBartModel", LSGMBartModel),
    "MBartForConditionalGeneration": ("LSGMBartForConditionalGeneration", LSGMBartForConditionalGeneration),
    "MBartForCausalLM": ("LSGMBartForCausalLM", LSGMBartForCausalLM),
    "MBartForQuestionAnswering": ("LSGMBartForQuestionAnswering", LSGMBartForQuestionAnswering),
    "MBartForSequenceClassification": ("LSGMBartForSequenceClassification", LSGMBartForSequenceClassification),
}
_MODEL_TYPE_DICT = {**{"LSG" + k: v for k, v in _MODEL_TYPE_DICT.items()}, **_MODEL_TYPE_DICT}

@dataclass
class FileArguments:
    """
    Arguments.
    """

    initial_model: str = field(
        metadata={"help": "Model to convert for long sequences"}
    )

    model_name: str = field(
        metadata={"help": "Name of saved model after conversion"}
    )

    max_sequence_length: int = field(
        default=4096,
        metadata={"help": "Max sequence length"}
    )

    architecture: str = field(
        default=None,
        metadata={
            "help": "Architecture or list of architectures (model specific, optional): " + ", ".join(_MODEL_TYPE_DICT.keys())}
    )

    random_global_init: bool = field(
        default=False,
        metadata={
            "help": "Randomly initialize global tokens (except the first one)"}
    )

    global_positional_stride: int = field(
        default=64,
        metadata={
            "help": "Positional stride of global tokens (copied from the original)"}
    )

    keep_first_global_token: bool = field(
        default=False,
        metadata={
            "help": "Do not replace an existing first global token (only used if initial model is already LSG type)"}
    )
    
    resize_lsg: bool = field(
        default=False,
        metadata={
            "help": "Only resize position embedding from a lsg model"}
    )

    seed: int = field(
        default=123,
        metadata={
            "help": "Set seed for random initialization"}
    )

def update_positions(model, max_pos, is_base=False):

    if is_base:
        processed_module = model
    else:
        processed_module = model.model

    # Encoder
    position_embeddings_weights = processed_module.encoder.embed_positions.weight.clone()
    current_max_position = position_embeddings_weights.size()[0]

    new_position_embeddings_weights = torch.cat([
        position_embeddings_weights[:2]] + 
        [position_embeddings_weights[2:] for _ in range(max_pos//current_max_position + 1)], 
        dim=0)[:max_pos + 2]

    #processed_module.encoder.embed_positions.position_ids = torch.arange(max_pos + 2, device=processed_module.encoder.embed_positions.position_ids.device).unsqueeze(0)
    processed_module.encoder.embed_positions.weight.data = new_position_embeddings_weights



    # Decoder
    position_embeddings_weights = processed_module.decoder.embed_positions.weight.clone()
    current_max_position = position_embeddings_weights.size()[0]

    new_position_embeddings_weights = torch.cat([
        position_embeddings_weights[:2]] + 
        [position_embeddings_weights[2:] for _ in range(max_pos//current_max_position + 1)], 
        dim=0)[:max_pos + 2]

    #processed_module.encoder.embed_positions.position_ids = torch.arange(max_pos + 2, device=processed_module.encoder.embed_positions.position_ids.device).unsqueeze(0)
    processed_module.decoder.embed_positions.weight.data = new_position_embeddings_weights

    return model

def order_positions(positions, stride):
    n, d = positions.size()
    if n % 512 != 0:
        if n > 512:
            positions = positions[:512*(n//512)]
        else:
            mean = positions.mean(dim=0, keepdim=True).expand(512 - n, -1)
            std = positions.std(dim=0, keepdim=True).expand(512 - n, -1)
            positions = torch.cat([positions, torch.normal(mean, std)], dim=0)
        n, d = positions.size()

    factor = n // 512
    positions = positions.reshape(-1, factor, d)[:, 0]
    positions = positions.reshape(-1, stride//factor, d).transpose(0, 1).reshape(-1, d)
    return positions

def update_global(model, bos, mask, stride, is_base=False, keep_first=False):
    
    if is_base:
        processed_module = model
    else:
        processed_module = model.model

    u = processed_module.shared.weight.clone()
    positions = processed_module.encoder.embed_positions.weight.clone()
    positions = order_positions(positions, stride)

    positions[0] += u[bos]
    positions[1:] += u[mask].unsqueeze(0)

    if keep_first:
        processed_module.encoder.global_embeddings.weight.data[1:] = positions[1:]
    else:
        processed_module.encoder.global_embeddings.weight.data = positions
    return model

def update_global_randomly(model, bos, stride, is_base=False, keep_first=False):

    import torch
    from torch.distributions.multivariate_normal import MultivariateNormal

    if is_base:
        processed_module = model
    else:
        processed_module = model.model

    u = processed_module.shared.weight.clone()
    cov = torch.cov(u.T)
    m = MultivariateNormal(u.mean(dim=0), cov)
    w = m.sample((512,))
    w[0] = u[bos]

    positions = processed_module.encoder.embed_positions.weight.clone()
    positions = order_positions(positions, stride)
    
    if keep_first:
        processed_module.encoder.global_embeddings.weight.data[1:] = (w + positions)[1:]
    else:
        processed_module.encoder.global_embeddings.weight.data = w + positions
    return model

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((FileArguments, ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    args = args[0]
    set_seed(args.seed)
    
    # Get config
    initial_config = AutoConfig.from_pretrained(args.initial_model, trust_remote_code=True, use_auth_token=True)
    if args.architecture is not None:
        model_type = args.architecture
        model_types = [model_type] if isinstance(model_type, str) else model_type
    else:
        # Get info from config
        model_types = initial_config.architectures

    # Get architecture
    if model_types is None:
        model_types = ["MBartForConditionalGeneration"]
        warnings.warn("Loaded architecture is None in config, will defaut to " + model_types[0])
    _architecture = _MODEL_TYPE_DICT.get(model_types[0], None)
    assert _architecture is not None, f"Provided/config architecture is wrong, make sure it is in {_MODEL_TYPE_DICT.keys()}" 
    
    _architecture, _model = _architecture
    _architectures = [_MODEL_TYPE_DICT[arc][0] for arc in model_types]

    # Load model
    config = LSGMBartConfig.from_pretrained(args.initial_model, architectures=_architectures)
    model = _model.from_pretrained(args.initial_model, use_auth_token=True, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.initial_model, use_auth_token=True)

    # Update tokenizer and config
    tokenizer.model_max_length = args.max_sequence_length
    tokenizer.init_kwargs['model_max_length'] = args.max_sequence_length

    max_pos = args.max_sequence_length
    model.config.max_position_embeddings = max_pos
    model.config._name_or_path = "ccdv/" + args.model_name

    # Hack because of architecture
    is_base = True if _architecture == "LSGMBartModel" else False

    # Check if it is LSG architecture
    is_lsg = True if vars(initial_config).get("base_model_prefix", None) == "lsg" else False
    if is_lsg and not args.resize_lsg:
        warnings.warn("LSG architecture detected, to resize positional embedding only, add --resize_lsg (won't affect global embedding)")
    if is_lsg and not args.keep_first_global_token:
        warnings.warn("LSG architecture detected, to keep the same first global token, add --keep_first_global_token")
    
    keep_first = False
    if args.keep_first_global_token:
        if is_lsg:
            keep_first = True
        else:
            warnings.warn("--keep_first_global_token won't be used if the initial model isn't a LSG model")

    # Update global embedding
    if not (is_lsg and args.resize_lsg):
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        mask_id = tokenizer.mask_token_id
        stride = args.global_positional_stride
        if args.random_global_init:
            model = update_global_randomly(model, bos_id, stride, is_base, keep_first)
        else:
            model = update_global(model, bos_id, mask_id, stride, is_base, keep_first)

    # Update positions
    model = update_positions(model, max_pos, is_base)
    
    model.save_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_name)

if __name__ == "__main__":
    main()