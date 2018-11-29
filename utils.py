# encoding = utf-8
# author = xy

from torch.utils import data
import torch


# word(list of list) -> index
def word2indexs(words, lang):

    def word2index(word_list):
        return [lang[word] if word in lang else lang['<unk>'] for word in word_list]

    return [word2index(word_list) for word_list in words]


# padding
def padding(words, max_len, pad_index=0):

    def padd(word_list):
        if len(word_list) > max_len:
            tmp = word_list[: max_len]
        else:
            tmp = word_list + [pad_index] * (max_len - len(word_list))
        return tmp

    results = [padd(word_list) for word_list in words]
    return results


# 构建 dataloader
def get_dataloader(dataset, batch_size, shuffle, drop_last):
    dataset = [torch.LongTensor(d) for d in dataset]
    dataset = data.TensorDataset(*dataset)
    data_iter = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter















