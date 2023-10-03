from torch.utils.data import Dataset
from TokenClassification._registry import register_dataset
from .load_data import load_CoNLL2003


@register_dataset
class CoNLL2003(Dataset):
    def __init__(self, data_path, is_train=True, labelTokenizer=None, label_type=2, tokenizer=None):
        """

        :param data_path: 对应的位置
        :param label_type: 0表示词性，1表示词块，2表示命名实体
        :param tokenizer: 用于编码的tokenizer
        """
        self.sentences, self.labels = load_CoNLL2003(data_path, label_type=label_type)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.labelTokenizer = labelTokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        label_enc = []
        input_ids = []
        attention_mask = []
        for word, tag in zip(self.sentences[idx], self.labels[idx]):
            items = self.tokenizer(word, add_special_tokens=False)
            input_ids.extend(items["input_ids"])
            attention_mask.extend(items["attention_mask"])
            label_enc.extend(self.labelTokenizer.label2id(tag, toTensor=False) * len(items["input_ids"]))
        label_enc += [0] + label_enc + [0]
        input_ids += [101] + input_ids + [102]
        attention_mask += [1] + attention_mask + [1]
        return {"input_ids": input_ids, "labels": label_enc, "attention_mask": attention_mask}
