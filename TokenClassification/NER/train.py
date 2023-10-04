from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from TokenClassification._registry import get_model, get_tokenizer, get_dataset, get_config
from Data.metrics import compute_metrics
from Config.base_config import baseConfig


def run(config=baseConfig()):
    model = get_model(config.model_config.model_name)
    label_tokenizer = get_tokenizer(config.tokenizer_config.label_tokenizer_name,
                                    **config.tokenizer_config.label_tokenizer_config)
    tokenizer = get_tokenizer(config.tokenizer_config.tokenizer_name,
                              **config.tokenizer_config.tokenizer_config)
    train_dataset = get_dataset(config.dataset_config.train_dataset_name,
                                labelTokenizer=label_tokenizer,
                                tokenizer=tokenizer,
                                **config.dataset_config.train_dataset_config)
    eval_dataset = get_dataset(config.dataset_config.val_dataset_name,
                               labelTokenizer=label_tokenizer,
                               tokenizer=tokenizer,
                               **config.dataset_config.val_dataset_config)
    data_collector = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                        **config.dataset_config.data_collator_config)
    training_args = TrainingArguments(**config.train_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
        args=training_args,  # training arguments, defined above 训练参数
        train_dataset=train_dataset,  # training dataset 训练集
        eval_dataset=eval_dataset,  # evaluation dataset 测试集
        compute_metrics=compute_metrics,
        data_collator=data_collector
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="meta_lstm_conll2003",
                        help="config的名字，可见Config文件对应类名")
    opt = parser.parse_args()
    config = get_config(opt.config_name)
    run(config)
    # tokenizer = get_tokenizer("bertTokenizerFast",
    #                           vocab_file=r"E:\LocalRepository\NaturalLanguageProcess\TokenizerFile\BertConll2003\vocab.txt",
    #                           tokenizer_file=r"E:\LocalRepository\NaturalLanguageProcess\TokenizerFile\BertConll2003\tokenizer.json",
    #                           do_lower_case=True)
    # print(tokenizer("MSK", add_special_tokens=False))
