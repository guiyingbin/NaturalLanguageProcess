import torch
import torch.nn as nn
from TokenClassification.NER.Utils.tools import log_sum_exp, argmax
from TokenClassification._registry import register_model
from torchcrf import CRF


class CRF_Custom(CRF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)


@register_model
class Bi_LSTM_CRF(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256, vocab_size=30552, target_size=9, drop_out=0.0,
                 num_layers=2):
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
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF_Custom(target_size, batch_first=True)
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
