import numpy as np
import torch
from TokenClassification._registry import register_tokenzier
from transformers import BertTokenizerFast
@register_tokenzier
class bertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register_tokenzier
class labelTokenizer(object):
    def __init__(self, label2id_dict, id2label_dict, pad_token_id=0, pad_token="O"):
        self.__label2id_dict = label2id_dict
        self.__id2label_dict = id2label_dict  # {v:k for k, v in label2id_dict.item()}
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
    def label2id(self, labels, toTensor=True):
        """

        :param labels:
        :param toTensor: 只有当长度相同时才可以填充
        :param padding: 填充为相同长度，如果padding为True, 那么一定可以toTensor
        :return:
        """
        output = []
        if isinstance(labels, str):
            output.append(self.__label2id_dict.get(labels, self.pad_token_id))
        elif isinstance(labels, (list, np.ndarray, torch.Tensor)):
            for label in labels:
                if isinstance(label, str):
                    output.append(self.__label2id_dict.get(label, self.pad_token_id))
                else:
                    output.append(self.label2id(label, False))
        else:
            raise ValueError("Not supported data type")
        if toTensor:
            return torch.LongTensor(output)
        else:
            return output

    def id2label(self, ids, toTensor=False):
        output = []
        if isinstance(ids, int):
            output.append(self.__id2label_dict.get(ids, self.pad_token))
        elif isinstance(ids, (list, np.ndarray, torch.Tensor)):
            for id in ids:
                if isinstance(id, int):
                    output.append(self.__id2label_dict.get(id, self.pad_token))
                else:
                    output.append(self.id2label(id, False))
        else:
            raise ValueError("Not supported data type")
        if toTensor:
            return torch.Tensor(output)
        else:
            return output


if __name__ == "__main__":
    label2id_dict = {"a": 0, "b": 1, "c": 2}
    id2label_dict = {v: k for k, v in label2id_dict.items()}
    tokenizer = labelTokenizer(label2id_dict, id2label_dict)
    a = [["a", "b", "c", "b", "c", "b"], ["a", "b", "c", "b"]]
    b = [[0, 1, 2, 3, -1, 2], [2, 1, 2, 3, 3]]
    print(tokenizer.label2id(a, toTensor=False))
    print(tokenizer.id2label(b, toTensor=False))
