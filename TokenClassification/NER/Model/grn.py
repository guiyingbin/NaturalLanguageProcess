from TokenClassification._registry import register_model
from TokenClassification.NER.Model.bi_lstm_crf import CRF_Custom
import torch.nn as nn
import torch.nn.functional as F
import torch


class ContextLayer(nn.Module):
    """
    source:https://github.com/microsoft/vert-papers/tree/master/papers/GRN-NER
    删除了mode，只保留了论文中的模式
    """

    def __init__(self, in_channels=1, out_channels=400, kernel_dim=400):
        # checked
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=(1, self.kernel_dim), padding=(0, 0))
        self.conv_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=(3, self.kernel_dim), padding=(1, 0))
        self.conv_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=(5, self.kernel_dim), padding=(2, 0))

    def forward(self, input):

        batch_size, max_seq, word_embed_size = input.size()

        # batch x 1 x max_seq x word_embed
        input_ = input.unsqueeze(1)

        input_1 = F.tanh(self.conv_1(input_))[:, :, :max_seq, :]
        input_3 = F.tanh(self.conv_3(input_))[:, :, :max_seq, :]
        input_5 = F.tanh(self.conv_5(input_))[:, :, :max_seq, :]
        # batch x out_channels x max_seq x 3
        pooling_input = torch.cat([input_1, input_3, input_5], 3)
        # batch x out_channels x max_seq x 1
        output = F.max_pool2d(pooling_input, kernel_size=(1, pooling_input.size(3)))

        # batch x out x max_seq -> batch x max_seq x out
        output = output.squeeze(3).transpose(1, 2)

        return output


class RelationLayer(nn.Module):
    """
    source:https://github.com/microsoft/vert-papers/tree/master/papers/GRN-NER
    删除了use_gpu,及删除自身的选项
    """

    def __init__(self, feature_dim=400):
        # checked
        super().__init__()

        self.feature_dim = feature_dim
        self.forget_gate_linear0 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        self.forget_gate_linear1 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)

    def forward(self, input, input_masks):
        """
        checked
        Get the context vector for each word
        :param input: features, batch x max_seq x embed/feature
        :param input_masks: binary mask matrix for the sentence words, batch x max_seq
        :param device: device to run the algorithm
        :return: context vector for each word, batch x max_seq x embed/feature
        """
        batch_size, max_seq_size, embed_size = input.size()
        sentence_lengths = torch.sum(input_masks, 1)

        assert self.feature_dim == embed_size

        # attention-based context information
        # batch x max_seq x embed --> batch x max_seq x max_seq x embed
        forget_gate0_linear = self.forget_gate_linear0(input)
        forget_gate1_linear = self.forget_gate_linear1(input)

        sigmoid_input = forget_gate0_linear.view(batch_size, max_seq_size, 1, embed_size) \
                            .expand(batch_size, max_seq_size, max_seq_size, embed_size) + forget_gate1_linear \
                            .view(batch_size, 1, max_seq_size, embed_size).expand(batch_size, max_seq_size,
                                                                                  max_seq_size, embed_size)

        # batch x max_seq x max_seq x embed
        forget_gate = torch.sigmoid(sigmoid_input)

        input_row_expanded = input.view(batch_size, 1, max_seq_size, embed_size) \
            .expand(batch_size, max_seq_size, max_seq_size, embed_size)

        forget_result = torch.mul(input_row_expanded, forget_gate)

        # start_t0 = time.time()
        selection_mask = input_masks.view(batch_size, max_seq_size, 1) \
            .mul(input_masks.view(batch_size, 1, max_seq_size))

        selection_mask = selection_mask.view(batch_size, max_seq_size, max_seq_size, 1) \
            .expand(batch_size, max_seq_size, max_seq_size, self.feature_dim)

        # batch x max_seq x max_seq x embed
        forget_result_masked = torch.mul(forget_result, selection_mask.float())

        # batch x max_seq x embed
        context_sumup = torch.sum(forget_result_masked, 2)

        # average
        context_vector = torch.div(context_sumup, sentence_lengths.view(batch_size, 1, 1)
                                   .expand(batch_size, max_seq_size, self.feature_dim).float())

        output_result = F.tanh(context_vector)

        return output_result


@register_model
class GRN(nn.Module):
    def __init__(self, embedding_size=128,
                 vocab_size=30552,
                 repre_out_chans=25,
                 repre_conv_kernel=3,
                 context_out_chans=25,
                 target_size=9,
                 drop_out=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.repre_out_chans = repre_out_chans
        self.context_out_chans = context_out_chans
        self.target_size = target_size
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)

        # represent layer, pooling layer 采用F.max_pool2d完成
        self.repre_conv = nn.Conv2d(in_channels=1, out_channels=repre_out_chans,
                                    kernel_size=(repre_conv_kernel, embedding_size),
                                    padding=(repre_conv_kernel // 2, 0))

        # context_layer
        self.context_layer = ContextLayer(in_channels=1, out_channels=context_out_chans,
                                          kernel_dim=embedding_size + repre_out_chans)

        # relation_layer 该层中有一步挺迷惑，即r_ij获取，所以采用的是源代码
        self.relation_layer = RelationLayer(feature_dim=context_out_chans)

        # CRF layer
        self.crf = CRF_Custom(target_size, batch_first=True)

        self.dropout = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(context_out_chans, target_size)

    def forward(self, input_ids, labels, attention_mask, char_input_ids, char_mask, char_len, input_len):
        """

        :param input_ids: word_level tokens, 和char_input_ids最长长度应保持一致
        :param labels:
        :param attention_mask:
        :param char_input_ids: char_level tokens
        :param char_mask:
        :param char_len:
        :param input_len:
        :return:
        """
        char_emb = self.emb_layer(char_input_ids)
        B, N, E = char_emb.shape
        char_emb = char_emb.unsqueeze(dim=1)  # (B, 1, N, E)
        c_char_emb = self.repre_conv(char_emb)  # (B, C1, N, 1) repre_out_chans is C1
        c_char_emb = F.max_pool2d(c_char_emb, kernel_size=(N, 1)).squeeze(dim=-1).squeeze(dim=-1)  # (B, C1)

        c_char_emb_temp = torch.zeros(B * N, self.repre_out_chans).to(c_char_emb.device)
        c_char_emb_temp[:B, :] = c_char_emb
        c_char_emb = c_char_emb_temp.view(B, N, self.repre_out_chans)

        word_emb = self.emb_layer(input_ids)  # (B, N, E)
        z_emb = torch.cat([c_char_emb, word_emb], dim=-1)
        x_emb = self.context_layer(z_emb) # (B, N, C2) context_out_chans is C2
        r_emb = self.relation_layer(x_emb, attention_mask)
        pred_logits = self.classifier(self.dropout(r_emb))
        loss = self.crf(pred_logits, labels, attention_mask) * (-1)
        return {"preds": pred_logits, "loss": loss}
