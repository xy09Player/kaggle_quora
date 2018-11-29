# encoding = utf-8
# author = xy


import pandas as pd
import numpy as np
import nltk


# 处理数据
# 1. 分词
# 2. 长度截断
def deal_data(data, max_len=100, is_train=True):
    df = pd.read_csv(data)
    questions = df['question_text'].values
    question_word_lists = [nltk.word_tokenize(q) for q in questions]
    question_word_list_len = [len(q) for q in question_word_lists]
    if is_train:
        target = df['target'].values
        question_os = []
        target_os = []
        for q, t, l in zip(question_word_lists, target, question_word_list_len):
            if l <= max_len:
                question_os.append(q)
                target_os.append(t)
        print('deal_data, retain data:%d/%d' % (len(question_os), len(questions)))
        return question_os, target_os

    else:
        question_os = question_word_lists
        return question_os


# 构建词表、embedding矩阵
def build_word_embedding(questions, glove_file):

    # 初始化embedding字典
    def get_matrixs(word, *nums):
        return word, np.asarray(nums, dtype='float32')
    embedding_dict = dict([get_matrixs(*o.split(' ')) for o in open(glove_file, 'r')])

    # 初始化词表
    word_set = set()
    for q in questions:
        for word in q:
            word_set.add(word)
    vocab_all_size = len(word_set)

    # 词表删选
    word_set = set()
    for q in questions:
        for word in q:
            if word in embedding_dict:
                word_set.add(word)
    vocab_size = len(word_set)

    print('words in pre-embedding, num:%d/%d, radio:%.4f' % (vocab_size, vocab_all_size, vocab_size/vocab_all_size))

    # 构建词表、embedding矩阵
    w2i = {'<pad>': 0}
    count = 1
    embedding = np.zeros([len(word_set)+2, 300])
    for word in word_set:
        if word not in w2i:
            w2i[word] = count
            embedding[count] = embedding_dict[word]
            count += 1
    w2i['<unk>'] = count
    assert len(w2i) == len(embedding)

    print('build_word_embedding,  vocab size:%d' % len(w2i))

    return w2i, embedding
