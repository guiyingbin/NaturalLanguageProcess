from torch.utils.data import Dataset
from TokenClassification._registry import register_dataset
from TokenClassification.NER.Data.load_data import load_CoNLL2003


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
        label_enc = [0] + label_enc + [0]
        input_ids = [101] + input_ids + [102]
        attention_mask = [1] + attention_mask + [1]
        return {"input_ids": input_ids, "labels": label_enc, "attention_mask": attention_mask}


@register_dataset
class meta_CoNLL2003(CoNLL2003):
    def __init__(self, data_path, is_train=True, labelTokenizer=None, label_type=2, tokenizer=None):
        super().__init__(data_path, is_train, labelTokenizer, label_type, tokenizer)
        self.n = len(self.sentences)

    def get_char_data(self, word, i, n_word):
        char_input_ids = []
        char_mask = []
        pad_id = 0 if len(word) == 1 and len(word)!=0 else 1
        # if i != n_word-1:
        temp_mask_ids = [-1] * (len(word) + 1)
        temp_mask_ids[0] = pad_id
        temp_mask_ids[-2] = pad_id
        char_mask.extend(temp_mask_ids)
        char_items = self.tokenizer(list(word + "_"), add_special_tokens=False)
        # else:
        #     temp_mask_ids = [-1] * (len(word))
        #     temp_mask_ids[0] = pad_id
        #     temp_mask_ids[-1] = pad_id
        #     char_mask.extend(temp_mask_ids)
        #     char_items = self.tokenizer(list(word), add_special_tokens=False)

        for char_token_list in char_items["input_ids"]:
            char_input_ids.extend(char_token_list)

        return char_input_ids, char_mask
    def decode_token(self, input_ids):
        word = []
        for ids in input_ids:
            if isinstance(ids, int):
                word.append(self.tokenizer.decode(ids))
            else:
                word.extend(self.decode_token(ids))
        return word

    def __getitem__(self, idx):
        label_enc = []
        char_input_ids = []
        char_mask = []  # 1表示start或者stop的位置， 0表示单个字符或者其他填充字符的位置
        input_ids = []
        attention_mask = []
        for i, (word, tag) in enumerate(zip(self.sentences[idx], self.labels[idx])):
            items = self.tokenizer(word, add_special_tokens=False)
            n_wordlist = self.decode_token(items["input_ids"])
            # 获取character level token
            for n_word in n_wordlist:
                temp_char_ids, temp_char_mask = self.get_char_data(n_word, i, len(self.sentences[idx]))
                char_mask.extend(temp_char_mask)
                char_input_ids.extend(temp_char_ids)

            input_ids.extend(items["input_ids"])
            attention_mask.extend(items["attention_mask"])
            label_enc.extend(self.labelTokenizer.label2id(tag, toTensor=False) * len(items["input_ids"]))

        char_input_ids = [101, 1035] + char_input_ids + [102]
        char_mask = [0, -1] + char_mask + [0]
        label_enc = [0] + label_enc + [0]
        input_ids = [101] + input_ids + [102]
        attention_mask = [1] + attention_mask + [1]
        return {"input_ids": input_ids,
                "labels": label_enc,
                "attention_mask": attention_mask,
                "char_input_ids": char_input_ids,
                "char_mask": char_mask,
                "char_len": len(char_input_ids),
                "input_len": len(input_ids)}
