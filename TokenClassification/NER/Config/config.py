from TokenClassification.NER.Config.base_config import baseConfig
from TokenClassification._registry import register_config
from easydict import EasyDict as edict


@register_config
class bi_lstm_crf_conll2003(baseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_config
class meta_lstm_conll2003(baseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_model_config()

    def set_model_config(self):
        self.model_config.model_name = "Meta_Bi_LSTM"
        self.model_config.model_config = edict(embedding_size=128, hidden_size=256,
                                               char_dim=16, word_dim=16, meta_dim=32,
                                               vocab_size=30552, target_size=9, drop_out=0.0,
                                               char_lstm_layers=2, word_lstm_layers=2, meta_lstm_layers=2,
                                               pretrained_embedding=None)
