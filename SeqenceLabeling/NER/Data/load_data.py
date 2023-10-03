"""
该部分主要用于放载入数据函数
"""

def load_CoNLL2003(data_path, label_type=2):
    """
    载入CoNLL2003数据，其格式为
    词  词性  词块  命名实体
    :param data_path: 原始的CoNLL2003数据
    :param label_type: 需要载入的label_type类型，0表示词性，1表示词块，2表示命名实体
    :return: sentences(list[list]), labels(list[list])
    """
    assert label_type in [0, 1, 2]
    sentences = []
    labels = []
    with open(data_path, "r", encoding="utf-8") as file:
        temp_sentence = []
        temp_label = []
        for line in file.readlines():
            line = line.strip()
            if line=="": #表示已经到一个句子末尾
                sentences.append(temp_sentence)
                labels.append(temp_label)
                temp_sentence = []
                temp_label = []
            else:
                word_labels_list = line.split(" ")
                temp_sentence.append(word_labels_list[0])
                temp_label.append(word_labels_list[label_type+1])
    return sentences, labels