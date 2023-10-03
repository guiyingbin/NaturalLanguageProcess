import torch
import torch.nn as nn
from SeqenceLabeling.NER.Utils.tools import log_sum_exp, argmax
from SeqenceLabeling._registry import register_model
from torchcrf import CRF


@register_model
class Bi_LSTM_CRF(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256, vocab_size=30552, target_size=9, drop_out=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(target_size, batch_first=True)
        # 采用该方式时，需要将CRF中_compute_normalizer中条件
        # score = torch.where(mask[i].unsqueeze(1), next_score, score)
        # 改为
        # score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
        # 因为输入的为attention_mask为long，但是现在torch.where后续版本并不支持long作为条件

    def forward_score(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embeddings)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    def forward(self, input_ids, labels, attention_mask):
        tag_scores = self.forward_score(input_ids)
        loss = self.crf(tag_scores, labels, attention_mask) * (-1)
        return {"preds": tag_scores, "loss": loss}
