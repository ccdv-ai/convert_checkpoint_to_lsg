from transformers import AutoConfig
from transformers.models.auto.modeling_auto import *
import json

from .albert.convert_albert_checkpoint import *
from .bart.convert_bart_checkpoint import *
from .barthez.convert_barthez_checkpoint import *
from .bert.convert_bert_checkpoint import *
from .camembert.convert_camembert_checkpoint import *
from .distilbert.convert_distilbert_checkpoint import *
from .electra.convert_electra_checkpoint import *
from .mbart.convert_mbart_checkpoint import *
from .pegasus.convert_pegasus_checkpoint import *
from .roberta.convert_roberta_checkpoint import *
from .xlm_roberta.convert_xlm_roberta_checkpoint import *

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

class LSGConverter():

    def __init__(
        self, 
        max_sequence_length=4096, 
        random_global_init=False, 
        global_positional_stride=64, 
        keep_first_global_token=False, 
        resize_lsg=False, 
        use_token_ids=True, 
        use_auth_token=False,
        seed=123,
        ):
        """
        max_sequence_length (int): new max sequence length
        random_global_init (bool): randomly initialize global tokens
        global_positional_stride (int): position stride between global tokens
        keep_first_global_token (bool): keep or replace the first global token (<s> + pos 0)
        resize_lsg (bool): only resize an existing LSG model
        use_token_ids (bool): use token_type_ids to build global tokens
        use_auth_token (bool): use HF auth token or not
        seed (int): seed
        """
        self.max_sequence_length = max_sequence_length
        self.random_global_init = random_global_init
        self.global_positional_stride = global_positional_stride
        self.keep_first_global_token = keep_first_global_token
        self.resize_lsg = resize_lsg
        self.use_token_ids = use_token_ids
        self.seed = seed

    def convert_from_pretrained(
        self, 
        model_name_or_path, 
        architecture=None, 
        use_auth_token=False,
        **model_kwargs
        ):
        """
        mode_name_or_path (str): path to the model to convert
        architecture (str): specific architecture (optional)
        model_kwargs: additional model args
        """

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, use_auth_token=use_auth_token)

        model_type = config.model_type
        model_kwargs = json.dumps(model_kwargs, indent=4)

        if model_type in _AUTH_MODELS.keys():
            converter = _AUTH_MODELS[model_type](
                initial_model=model_name_or_path, 
                model_name=model_name_or_path, 
                max_sequence_length=self.max_sequence_length, 
                architecture=architecture, 
                random_global_init=self.random_global_init, 
                global_positional_stride=self.global_positional_stride, 
                keep_first_global_token=self.keep_first_global_token, 
                resize_lsg=self.resize_lsg, 
                model_kwargs=model_kwargs, 
                use_token_ids=self.use_token_ids,
                use_auth_token=use_auth_token,
                config=config,
                save_model=False,
                seed=self.seed
                )
            return converter.process()

"""
model, tokenizer = LSGConverter().convert_from_pretrained("bert-base-uncased", num_global_tokens=15)
print(model)
print(model.config)
"""