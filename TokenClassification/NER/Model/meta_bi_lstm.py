import torch
import torch.nn as nn
from TokenClassification._registry import register_model

@register_model
class Meta_Bi_LSTM(nn.Module):
    """
    paper: Morphosyntactic Tagging with a Meta-BiLSTM Model over Context Sensitive Token Encodings
    https://arxiv.org/pdf/1805.08237.pdf
    """
    def __init__(self, embedding_size=128, hidden_size=256,
                 char_dim=16, word_dim=16, meta_dim=32,
                 vocab_size=30552, target_size=9, drop_out=0.0,
                 char_lstm_layers=2, word_lstm_layers=2, meta_lstm_layers=2,
                 pretrained_embedding=None):
        super().__init__()
        self.hidden_size = hidden_size
        if pretrained_embedding is None:
            self.pre_emb = torch.rand((1, 1, embedding_size))
        else:
            self.pre_emb = pretrained_embedding
        self.target_size = target_size
        self.emb = nn.Embedding(vocab_size, embedding_size)
        # character model
        self.char_model_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                       batch_first=True, bidirectional=True,
                                       num_layers=char_lstm_layers,
                                       dropout=drop_out)
        self.char_model_mlp = nn.Linear(4, char_dim)
        self.char_model_classifer = nn.Sequential(nn.Linear(char_dim, target_size),
                                                  nn.Softmax())
        # word model
        self.word_model_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                       batch_first=True, bidirectional=True,
                                       num_layers=word_lstm_layers,
                                       dropout=drop_out)
        self.word_model_mlp = nn.Linear(2*hidden_size, word_dim)
        self.word_model_classifier = nn.Sequential(nn.Linear(word_dim, target_size),
                                                  nn.Softmax())

        # meta model
        self.meta_model_lstm = nn.LSTM(input_size=word_dim+char_dim, hidden_size=meta_dim,
                                       batch_first=True, bidirectional=True,
                                       num_layers=meta_lstm_layers,
                                       dropout=drop_out)
        self.meta_model_mlp = nn.Sequential(nn.Linear(meta_dim*2, target_size),
                                            nn.Softmax())

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, char_logits, word_logits, meta_logits, labels, attention_mask):
        char_loss = self.criterion(char_logits, labels)*attention_mask
        word_loss = self.criterion(word_logits, labels)*attention_mask
        meta_loss = self.criterion(meta_logits, labels)*attention_mask
        return char_loss.mean()+word_loss.mean()+meta_loss.mean()
    def forward(self, input_ids, labels, attention_mask):
        embed = self.emb(input_ids)
        # character 部分
        f_b_c, _ = self.char_model_lstm(embed)
        B, N, H = f_b_c.shape
        f_b_c = f_b_c.view(B, N, 2, H // 2)
        f, b = f_b_c[:, :, 0, :], f_b_c[:, :, 1, :]
        f_f, f_l, b_f, b_l = f[:, :, 0], f[:, :, -1], b[:, :, 0], b[:, :, -1]
        n_f_b = torch.stack([f_f, f_l, b_f, b_l], dim=-1)
        m_char = self.char_model_mlp(n_f_b)
        char_pred_logits = self.char_model_classifer(m_char)

        # word 部分
        word_embed = embed+self.pre_emb
        f_b_w, _ = self.word_model_lstm(word_embed)
        m_word = self.word_model_mlp(f_b_w)
        word_pred_logits = self.word_model_classifier(m_word)

        # meta 部分
        c_w = torch.concat([m_char.detach(), m_word.detach()], dim=-1)
        f_b_m, _ = self.meta_model_lstm(c_w)
        meta_pred_logits = self.meta_model_mlp(f_b_m)
        loss = self.compute_loss(char_pred_logits.view(-1, self.target_size),
                                 word_pred_logits.view(-1, self.target_size),
                                 meta_pred_logits.view(-1, self.target_size),
                                 labels.view(-1), attention_mask.view(-1))
        return {"loss":loss, "predictions":meta_pred_logits}

if __name__ == "__main__":
    loss = nn.CrossEntropyLoss(reduction="none")
    a = torch.rand(4, 66, 9)
    b = torch.randn(4, 66).clip(0, 9).long()
    print(loss(a.view(-1, 9), b.view(-1)))