# encoding = utf-8
# author = xy


import pandas as pd
import numpy as np
import nltk
import re
from tool import contraction_mapping
from tool import specials_d
from tool import punct
from tool import specials_c
from tool import mispell_dict


def build_vocab(questions):
    vocab = set()
    for question in questions:
        for word in question.split():
            vocab.add(word)



# 预处理

## 小写

## 缩写替换

## 标点切割

def pre_data(sentence_list):

    result = []
    for i in sentence_list:
        i = re.sub(r'\s+', ' ', i)

        # 小写
        i = i.lower()

        # 缩写词替换
        for s in specials_d:
            i = i.replace(s, "'")
        for word in contraction_mapping.keys():
            i = i.replace(word, contraction_mapping[word])

        # 错误词替换
        for s in specials_c:
            i = i.replace(s, specials_c[s])
        for word in mispell_dict.keys():
            i = i.replace(word, mispell_dict[word])

        i = re.sub(r'\s+', ' ', i)
        i = nltk.word_tokenize(i)

        result.append(i)

    return result


# 处理数据
# 1. 分词
# 2. 长度截断
def deal_data(data, max_len=100, is_train=True):
    df = pd.read_csv(data)
    questions = df['question_text'].values
    question_word_lists = pre_data(questions)
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

    # 小写化字典
    embedding_dict_tmp = {}
    for word in embedding_dict.keys():
        if word.lower() in embedding_dict:
            embedding_dict_tmp[word.lower()] = embedding_dict[word.lower()]
        else:
            embedding_dict_tmp[word.lower()] = embedding_dict[word]
    print('lower embedding_dict: %d/%d' % (len(embedding_dict_tmp), len(embedding_dict)))
    embedding_dict = embedding_dict_tmp

    # 初始化词表
    vocab = {}
    for q in questions:
        for word in q:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1

    # 检查覆盖率、词表删选
    known_num = 0
    all_num = 0
    word_set = []
    for word in vocab.keys():
        if word in embedding_dict:
            known_num += vocab[word]
            word_set.append(word)
        all_num += vocab[word]

    print('words in pre-embedding, num:%d/%d, radio:%.4f' % (len(word_set), len(vocab), len(word_set)/len(vocab)))
    print('known words in all text:%.4f' % (known_num/all_num))

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
