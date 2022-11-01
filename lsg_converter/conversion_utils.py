
from transformers import AutoTokenizer
import json
import warnings
import torch
import sys 

class ConversionScript():

    _ARCHITECTURE_TYPE_DICT = {}
    _ARCHITECTURE_TYPE_DICT = {**{"LSG" + k: v for k, v in _ARCHITECTURE_TYPE_DICT.items()}, **_ARCHITECTURE_TYPE_DICT}
    _BASE_ARCHITECTURE_TYPE = None
    _DEFAULT_ARCHITECTURE_TYPE = None
    _CONFIG_MODULE = None

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
        use_token_ids,
        use_auth_token,
        config,
        save_model,
        seed
        ):
        
        self.initial_model = initial_model
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.architecture = architecture
        self.random_global_init = random_global_init
        self.global_positional_stride = global_positional_stride
        self.keep_first_global_token = keep_first_global_token
        self.resize_lsg = resize_lsg
        self.model_kwargs = model_kwargs
        self.use_token_ids = use_token_ids
        self.use_auth_token = use_auth_token
        self.config = config
        self.save_model = save_model

    def save(self, model, tokenizer):

        model.save_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_name)

    def process(self):
        
        (lsg_architecture, lsg_model), initial_architecture = self.get_architecture()
        is_base_architecture, is_lsg, keep_first_global = self.get_additional_params(lsg_architecture, initial_architecture)
        model, tokenizer = self.get_model(lsg_architecture, lsg_model)
        model, tokenizer = self.update_config(model, tokenizer)

        # Get the module prefix to update
        module_prefix = self.get_module(model, is_base_architecture)

        # Update global embedding
        if not (is_lsg and self.resize_lsg):
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
            bos_id = bos_id if bos_id is not None else model.config.bos_token_id
            mask_id = tokenizer.mask_token_id
            if self.random_global_init:
                self.update_global_randomly(module_prefix, bos_id, self.global_positional_stride, keep_first_global)
            else:
                self.update_global(module_prefix, bos_id, mask_id, self.global_positional_stride, keep_first_global)

        # Update positional
        self.update_positions(module_prefix, self.max_sequence_length)

        # For Pegasus
        self.update_positions_with_model(model, self.max_sequence_length)

        if self.save_model:
            self.save(model, tokenizer)

        return model, tokenizer

    def get_architecture(self):
        if self.architecture is not None:
            return self.validate_architecture(self.architecture)

        architectures = self.config.architectures
        if architectures is not None:
            architecture = architectures if isinstance(architectures, str) else architectures[0]
            return self.validate_architecture(architecture)

        return self.validate_architecture(self._DEFAULT_ARCHITECTURE_TYPE)

    def validate_architecture(self, architecture):
        _architecture = self._ARCHITECTURE_TYPE_DICT.get(architecture, None)

        s = "\n * " + "\n * ".join([k for k in self._ARCHITECTURE_TYPE_DICT.keys()])
        assert _architecture is not None, f"Provided/config architecture is wrong, make sure it is in: {s}"
        return _architecture, architecture

    def get_model(self, lsg_architecture, lsg_model):
        config = self._CONFIG_MODULE.from_pretrained(
            self.initial_model, 
            architectures=lsg_architecture, 
            trust_remote_code=True, 
            use_auth_token=self.use_auth_token,
            **json.loads(self.model_kwargs.replace("'", "\""))
            )
        model = lsg_model.from_pretrained(self.initial_model, use_auth_token=self.use_auth_token, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.initial_model, use_auth_token=self.use_auth_token, trust_remote_code=True)
        return model, tokenizer

    def update_config(self, model, tokenizer):

        # Update tokenizer and config
        tokenizer.model_max_length = self.max_sequence_length
        tokenizer.init_kwargs['model_max_length'] = self.max_sequence_length

        max_pos = self.max_sequence_length
        model.config.max_position_embeddings = max_pos + self._DEFAULT_CONFIG_POSITIONAL_OFFSET
        model.config._name_or_path = self.model_name
        return model, tokenizer

    def get_additional_params(self, _architecture, initial_architecture):

        # Hack because of architecture
        is_base_architecture = True if _architecture in [self._BASE_ARCHITECTURE_TYPE, "LSG" + self._BASE_ARCHITECTURE_TYPE] else False

        # Check if it is LSG architecture
        if vars(self.config).get("base_model_prefix", None) == "lsg" or "LSG" in initial_architecture:
            is_lsg_architecture = True
        else: 
            is_lsg_architecture = False

        if is_lsg_architecture and not self.resize_lsg:
            warnings.warn("LSG architecture detected, to resize positional embedding only, add --resize_lsg (won't affect global embedding)")
        if is_lsg_architecture and not self.keep_first_global_token and not self.resize_lsg:
            warnings.warn("LSG architecture detected, to keep the same first global token, add --keep_first_global_token")

        keep_first = False
        if self.keep_first_global_token:
            if is_lsg_architecture:
                keep_first = True
            else:
                warnings.warn("--keep_first_global_token won't be used if the initial model isn't a LSG model")
        return is_base_architecture, is_lsg_architecture, keep_first

    def get_module(self, model, is_base_architecture):
        if is_base_architecture:
            return
        return

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):
        pass

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):
        pass

    def update_positions(self, module_prefix, max_pos):
        pass
    
    def update_positions_with_model(self, model, max_pos):
        pass

    def order_positions(self, positions, stride):
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

    def run_test(self):
        pass
    
    def run_models(self, lsg_path, max_length, hidden_size, text, auto_map, gradient_checkpointing=True, is_encoder_decoder=False):

        from transformers import AutoTokenizer, AutoConfig, AutoModel, pipeline
        from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering
        from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(lsg_path)
        
        long_text = text * 200

        for name in auto_map.keys():

            if name == "AutoConfig":
                continue

            model = getattr(sys.modules["transformers"], name)
            print("\n\n" + "="*5 + " " + name + " " + "="*5 + "\n")
            model = model.from_pretrained(lsg_path, trust_remote_code=True, is_decoder="Causal" in name)
            
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()

            if "QuestionAnswering" in name:
                tokens = tokenizer("context", long_text, return_tensors="pt", truncation=True)
                inputs_embeds = torch.randn(1, max_length, hidden_size)
            elif "MultipleChoice" in name:
                num_choices = 4
                tokens = tokenizer([long_text]*num_choices, return_tensors="pt", truncation=True)
                tokens = {k: v.reshape(1, num_choices, -1) for k, v in tokens.items()}
                inputs_embeds = torch.randn(1, num_choices, max_length//4, hidden_size)
            else:
                tokens = tokenizer(long_text, return_tensors="pt", truncation=True)
                inputs_embeds = torch.randn(1, max_length, hidden_size)

            if model.config.model_type != "pegasus":
                model(**tokens)
                
            if not is_encoder_decoder:
                model(inputs_embeds=inputs_embeds)
            elif "decoder_input_ids" in model.forward.__code__.co_varnames:
                decoder_input_ids = tokens.input_ids[:, :256]
                if "SequenceClassification" not in name:
                    model(**tokens, decoder_input_ids=decoder_input_ids)

