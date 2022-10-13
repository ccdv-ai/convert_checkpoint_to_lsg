import os
import sys
from dataclasses import dataclass, field

import warnings

from transformers import (
    AutoConfig,
    HfArgumentParser,
    set_seed,
)

from convert_models.albert.convert_albert_checkpoint import *
from convert_models.bart.convert_bart_checkpoint import *
from convert_models.barthez.convert_barthez_checkpoint import *
from convert_models.bert.convert_bert_checkpoint import *
from convert_models.camembert.convert_camembert_checkpoint import *
from convert_models.distilbert.convert_distilbert_checkpoint import *
from convert_models.electra.convert_electra_checkpoint import *
from convert_models.mbart.convert_mbart_checkpoint import *
from convert_models.pegasus.convert_pegasus_checkpoint import *
from convert_models.roberta.convert_roberta_checkpoint import *
from convert_models.xlm_roberta.convert_xlm_roberta_checkpoint import *


_AUTH_MODELS = {
    "albert": AlbertConversionScript,
    "bart": BartConversionScript,
    "barthez": BarthezConversionScript,
    "bert": BertConversionScript,
    "camembert": CamembertConversionScript,
    "distilbert": DistilBertConversionScript,
    "electra": ElectraConversionScript,
    "mbart": MBartConversionScript,
    "pegasus": PegasusConversionScript,
    "roberta": RobertaConversionScript,
    "xlm-roberta": XLMRobertaConversionScript,
}


@dataclass
class FileArguments:
    """
    Arguments.
    """

    initial_model: str = field(
        metadata={"help": "Model to convert to its LSG variant"}
    )

    model_name: str = field(
        metadata={"help": "Name/path of the newly created model"}
    )

    max_sequence_length: int = field(
        default=4096,
        metadata={"help": "Max sequence length"}
    )

    architecture: str = field(
        default=None,
        metadata={
            "help": "Architecture (model specific, optional, e.g BartForConditionalGeneration)"}
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
            "help": "Only resize the positional embedding from a LSG model (skip global tokens)"}
    )

    model_kwargs: Optional[str] = field(
        default="{}",
        metadata={
            "help": "Model kwargs, ex: \"{'sparsity_type': 'none', 'mask_first_token': true}\""
        },
    )

    use_token_ids: bool = field(
        default=True,
        metadata={
            "help": "Add token_type_ids (0) to global embeddings if the model allows it"}
    )

    run_test: bool = field(
        default=False,
        metadata={
            "help": "Test the architecture of the new model"}
    )

    seed: int = field(
        default=123,
        metadata={
            "help": "Set seed for random initialization"}
    )

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
    config = AutoConfig.from_pretrained(args.initial_model, trust_remote_code=True, use_auth_token=True)
    model_type = config.model_type

    if model_type in _AUTH_MODELS.keys():
        converter = _AUTH_MODELS[model_type](
            initial_model=args.initial_model, 
            model_name=args.model_name, 
            max_sequence_length=args.max_sequence_length, 
            architecture=args.architecture, 
            random_global_init=args.random_global_init, 
            global_positional_stride=args.global_positional_stride, 
            keep_first_global_token=args.keep_first_global_token, 
            resize_lsg=args.resize_lsg, 
            model_kwargs=args.model_kwargs, 
            use_token_ids=args.use_token_ids,
            config=config,
            seed=args.seed
            )
        converter.process()

        if args.run_test:
            converter.run_test()
        
    else:
        s = "\n * " + "\n * ".join([k for k in _AUTH_MODELS.keys()])
        warnings.warn(f"Model type <{model_type}> can not be handled by this script. Model type must be one of: {s}")
    
    
if __name__ == "__main__":
    main()