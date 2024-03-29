from TokenClassification.NER.Config.base_config import baseConfig
from TokenClassification._registry import register_config
from easydict import EasyDict as edict
from transformers.utils import PaddingStrategy


@register_config
class bi_lstm_crf_conll2003(baseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_config
class meta_lstm_conll2003(baseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_model_config()
        self.set_data_config()
        self.set_tokenizer_config()

    def set_model_config(self):
        self.model_config.model_name = "Meta_Bi_LSTM"
        self.model_config.model_config = edict(embedding_size=128, hidden_size=256,
                                               char_dim=16, word_dim=16, meta_dim=32,
                                               vocab_size=30552, target_size=9, drop_out=0.0,
                                               char_lstm_layers=2, word_lstm_layers=2, meta_lstm_layers=2,
                                               pretrained_embedding=None)

    def set_data_config(self):
        self.dataset_config.train_dataset_name = "meta_CoNLL2003"
        self.dataset_config.val_dataset_name = "meta_CoNLL2003"

    def set_tokenizer_config(self):
        self.tokenizer_config.tokenizer_name = "bertTokenizerFast_custom"


@register_config
class grn_conll2003(meta_lstm_conll2003):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_model_config(self):
        self.model_config.model_name = "GRN"
        self.model_config.model_config = edict(embedding_size=128,
                                               vocab_size=30552,
                                               repre_out_chans=25,
                                               repre_conv_kernel=3,
                                               context_out_chans=25,
                                               target_size=9,
                                               drop_out=0.0)

    def set_data_config(self):
        self.dataset_config.train_dataset_name = "meta_CoNLL2003"
        self.dataset_config.val_dataset_name = "meta_CoNLL2003"
        self.dataset_config.data_collator_config = edict(label_pad_token_id=0,
                                                         max_length=1024,
                                                         padding="max_length")
