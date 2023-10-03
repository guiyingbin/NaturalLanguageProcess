from easydict import EasyDict as edict

_base_train_config = edict(output_dir=r'E:\LocalRepository\NaturalLanguageProcess',  # output directory 结果输出地址
                           num_train_epochs=1000,  # total # of training epochs 训练总批次
                           per_device_train_batch_size=5,  # batch size per device during training 训练批大小
                           per_device_eval_batch_size=5,  # batch size for evaluation 评估批大小
                           logging_dir=r'E:\LocalRepository\NaturalLanguageProcess\logs',
                           # directory for storing logs 日志存储位置
                           report_to=["wandb"],  # 将结果保存到哪
                           run_name="test",  # wandb
                           optim="adamw_torch",
                           # adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor
                           learning_rate=1e-3,  # 学习率
                           save_steps=False,  # 不保存检查点
                           evaluation_strategy="epoch",  # 验证策略
                           gradient_accumulation_steps=1,  # 梯度累积step
                           warmup_steps=500,  # number of warmup steps for learning rate scheduler 预热学习率调整器步数
                           weight_decay=0.01,
                           lr_scheduler_type="constant_with_warmup",
                           # linear, cosine, cosine_with_restarts, polynomial, constant,
                           # constant_with_warmup, inverse_sqrt, reduce_lr_on_plateau
                           seed=42,  # 随机种子
                           bf16=False,  # 是否采用bf16训练
                           fp16=False,  # 是否采用fp16训练
                           metric_for_best_model="acc")

_base_dataset_config = edict(train_dataset_name="CoNLL2003",
                             train_dataset_config=edict(
                                 data_path=r"E:\LocalRepository\Data\NLP\NamedEntity\CoNLL2003\eng.train.txt",
                                 is_train=True,
                                 label_type=2),
                             val_dataset_name="CoNLL2003",
                             val_dataset_config=edict(
                                 data_path=r"E:\LocalRepository\Data\NLP\NamedEntity\CoNLL2003\eng.train.txt",
                                 is_train=True,
                                 label_type=2),
                             data_collator_config=edict(label_pad_token_id=0,
                                                        max_length=128))

_base_model_config = edict(model_name="Bi_LSTM_CRF",
                           model_config=edict(embedding_size=128,
                                              hidden_size=256,
                                              vocab_size=30552,
                                              target_size=9,
                                              drop_out=0.0))

_base_tokenizer_config = edict(label_tokenizer_name="labelTokenizer",
                               label_tokenizer_config=edict(label2id_dict={"B-LOC": 5,
                                                                           "B-MISC": 7,
                                                                           "B-ORG": 3,
                                                                           "B-PER": 1,
                                                                           "I-LOC": 6,
                                                                           "I-MISC": 8,
                                                                           "I-ORG": 4,
                                                                           "I-PER": 2,
                                                                           "O": 0},
                                                            id2label_dict={"0": "O",
                                                                           "1": "B-PER",
                                                                           "2": "I-PER",
                                                                           "3": "B-ORG",
                                                                           "4": "I-ORG",
                                                                           "5": "B-LOC",
                                                                           "6": "I-LOC",
                                                                           "7": "B-MISC",
                                                                           "8": "I-MISC"}),
                               tokenizer_name="bertTokenizerFast",
                               tokenizer_config=edict(
                                   vocab_file=r"E:\LocalRepository\NaturalLanguageProcess\TokenizerFile\BertConll2003\vocab.txt",
                                   tokenizer_file=r"E:\LocalRepository\NaturalLanguageProcess\TokenizerFile\BertConll2003\tokenizer.json",
                                   do_lower_case=True)
                               )


class baseConfig(object):
    __base_dataset_config = _base_dataset_config
    __base_train_config = _base_train_config
    __base_model_config = _base_model_config
    __base_tokenizer_config = _base_tokenizer_config

    def __init__(self, *args, **kwargs):
        pass

    @property
    def dataset_config(self):
        return self.__base_dataset_config

    @property
    def train_config(self):
        return self.__base_train_config

    @property
    def model_config(self):
        return self.__base_model_config

    @property
    def tokenizer_config(self):
        return self.__base_tokenizer_config


if __name__ == "__main__":
    arg = baseConfig()
    print(arg.train_config)
