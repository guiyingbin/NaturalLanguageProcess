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

        if pretrained_embedding is None:
            self.pre_emb = torch.rand((1, 1, embedding_size))
        else:
            self.pre_emb = pretrained_embedding
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.emb_size = embedding_size
        self.emb = nn.Embedding(vocab_size, embedding_size)
        # character model
        self.char_model_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                       batch_first=True, bidirectional=True,
                                       num_layers=char_lstm_layers,
                                       dropout=drop_out)
        self.char_model_mlp = nn.Linear(4 * hidden_size, char_dim)
        self.char_model_classifer = nn.Sequential(nn.Linear(char_dim, target_size),
                                                  nn.Softmax())
        # word model
        self.word_model_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                       batch_first=True, bidirectional=True,
                                       num_layers=word_lstm_layers,
                                       dropout=drop_out)
        self.word_model_mlp = nn.Linear(2 * hidden_size, word_dim)
        self.word_model_classifier = nn.Sequential(nn.Linear(word_dim, target_size),
                                                   nn.Softmax())

        # meta model
        self.meta_model_lstm = nn.LSTM(input_size=word_dim + char_dim, hidden_size=meta_dim,
                                       batch_first=True, bidirectional=True,
                                       num_layers=meta_lstm_layers,
                                       dropout=drop_out)
        self.meta_model_mlp = nn.Sequential(nn.Linear(meta_dim * 2, target_size),
                                            nn.Softmax())

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, char_logits, word_logits, meta_logits, labels, attention_mask):
        char_loss = self.criterion(char_logits, labels) * attention_mask
        word_loss = self.criterion(word_logits, labels) * attention_mask
        meta_loss = self.criterion(meta_logits, labels) * attention_mask
        return char_loss.mean() + word_loss.mean() + meta_loss.mean()

    def forward_char_model(self, char_input_ids, char_mask, attention_mask, char_lens, input_len):
        B, N = attention_mask.shape
        n_f_b_list = []
        for i in range(B):
            n_word = input_len[i]
            difference = N - n_word
            n_char = char_lens[i]

            t_char_id = torch.cat([char_input_ids[i, :n_char],
                                   torch.zeros(size=(difference,), requires_grad=False, dtype=torch.long)],
                                  dim=0).unsqueeze(0)
            t_char_mask = torch.cat([char_mask[i, :n_char],
                                     torch.zeros(size=(difference,), requires_grad=False, dtype=torch.long)],
                                    dim=0).unsqueeze(0)
            char_embed = self.emb(t_char_id)
            f_b_c, _ = self.char_model_lstm(char_embed)  # [B, N, 2*h]
            start_stop_ids = torch.where(t_char_mask == 1)  # 抽取每个word的开头和结尾位置
            other_ids = torch.where(t_char_mask == 0)
            f_b_c_1 = f_b_c[start_stop_ids[0], start_stop_ids[1], :]  # f_b_c_1 [C1, 2*h]
            f_b_c_1 = f_b_c_1.reshape(-1, self.hidden_size * 4)  # f_b_c_1 [C1, 4*h]
            # other_ids由于找的都是单个字符，所以start和stop是同一位置，需要将向量维度*2拼接
            f_b_c_2 = torch.concat([f_b_c[other_ids[0], other_ids[1], :],
                                    f_b_c[other_ids[0], other_ids[1], :]], dim=-1)  # f_b_c_2[C2, 4*h]

            n_f_b_list.append(torch.concat([f_b_c_1, f_b_c_2], dim=0).view(1, N, self.hidden_size * 4))
        m_char = self.char_model_mlp(torch.cat(n_f_b_list, dim=0))
        char_pred_logits = self.char_model_classifer(m_char)
        return m_char, char_pred_logits

    def forward_word_model(self, input_ids):
        word_embed = self.emb(input_ids)  # 这里char和word-level应该是不同embed,这里是简写
        B, N, _ = word_embed.shape
        word_embed = word_embed + self.pre_emb
        f_b_w, _ = self.word_model_lstm(word_embed)
        m_word = self.word_model_mlp(f_b_w)
        word_pred_logits = self.word_model_classifier(m_word)
        return m_word, word_pred_logits

    def forward(self, input_ids, labels, attention_mask, char_input_ids, char_mask, char_len, input_len):
        """

        :param input_ids: shape [B, N]
        :param labels: shape [B, N]
        :param attention_mask: shape [B, N]
        :param char_input_ids: shape [B, N]
        :param char_mask: shape [B, N]
        :return:
        """

        # character 部分
        m_char, char_pred_logits = self.forward_char_model(char_input_ids, char_mask,
                                                           attention_mask, char_len, input_len)

        # word 部分
        m_word, word_pred_logits = self.forward_word_model(input_ids)

        # meta 部分
        c_w = torch.concat([m_char.detach(), m_word.detach()], dim=-1)
        f_b_m, _ = self.meta_model_lstm(c_w)
        meta_pred_logits = self.meta_model_mlp(f_b_m)
        loss = self.compute_loss(char_pred_logits.view(-1, self.target_size),
                                 word_pred_logits.view(-1, self.target_size),
                                 meta_pred_logits.view(-1, self.target_size),
                                 labels.view(-1), attention_mask.view(-1))
        return {"loss": loss, "predictions": meta_pred_logits}


if __name__ == "__main__":
    loss = nn.CrossEntropyLoss(reduction="none")
    a = torch.rand(4, 66, 9)
    b = torch.randn(4, 66).clip(0, 9).long()
    print(loss(a.view(-1, 9), b.view(-1)))
