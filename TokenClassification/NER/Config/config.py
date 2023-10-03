from TokenClassification.NER.Config.base_config import baseConfig
from TokenClassification._registry import register_config

@register_config
class bi_lstm_crf_conll2003(baseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


