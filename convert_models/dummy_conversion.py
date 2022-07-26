from .conversion_utils import ConversionScript


class DummyConversionScript(ConversionScript):

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
            return
        return

    def update_global_randomly(self, module_prefix, bos_id, stride, keep_first_global):
        pass

    def update_global(self, module_prefix, bos_id, mask_id, stride, keep_first_global):
        pass

    def update_positions(self, module_prefix, max_pos):
        pass