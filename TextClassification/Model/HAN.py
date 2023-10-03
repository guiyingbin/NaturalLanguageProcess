"""
source:https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from TextClassification.Utils.layers import WordAttention, SentenceAttention


class HAN(nn.Module):
    def __init__(self, n_class, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout=0.1):
        super().__init__()
        self.head = nn.Linear(2 * sentence_rnn_size, n_class)
        self.word_attention = WordAttention(vocab_size=vocab_size,
                                            emb_size=emb_size,
                                            word_rnn_size=word_rnn_size,
                                            word_rnn_layers=word_rnn_layers,
                                            word_att_size=word_att_size,
                                            dropout=dropout)
        self.sentence_attention = SentenceAttention(word_rnn_size=word_rnn_size,
                                                    sentence_rnn_size=sentence_rnn_size,
                                                    sentence_rnn_layers=sentence_rnn_layers,
                                                    sentence_att_size=sentence_att_size,
                                                    dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(documents,
                                                lengths=sentences_per_document.tolist(),
                                                batch_first=True,
                                                enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence,
                                                         lengths=sentences_per_document.tolist(),
                                                         batch_first=True,
                                                         enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(packed_sentences.data,
                                                     packed_words_per_sentence.data)  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)
        documents_vector, sentences_alphas = self.sentence_attention(sentences, packed_sentences)

        word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas,
                                                            batch_sizes=packed_sentences.batch_sizes,
                                                            sorted_indices=packed_sentences.sorted_indices,
                                                            unsorted_indices=packed_sentences.unsorted_indices),
                                             batch_first=True)
        preds = self.head(documents_vector)
        return preds, sentences_alphas, word_alphas


if __name__ == "__main__":
    han = HAN(n_class=3, vocab_size=6624, emb_size=192, word_rnn_size=192, word_rnn_layers=2,
              word_att_size=384, sentence_rnn_size=192, sentence_att_size=384, sentence_rnn_layers=2,
              dropout=0.1)
    documents = torch.LongTensor([
        [
            [1, 2, 354, 1323, 3423, 2232, 342, 423, 342, 342, 342, 23, 0, 0],
            [1, 2, 354, 1323, 3423, 2232, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [1, 2, 354, 1323, 3423, 2232, 342, 423, 342, 342, 342, 23, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]]) # (2, 2, 14)
    sentence_per_document = torch.LongTensor([2, 1])
    word_per_sentence = torch.LongTensor([[12, 6],
                                          [12, 0]])
    v, _, _ = han(documents, sentence_per_document, word_per_sentence)
    print(v.shape)
