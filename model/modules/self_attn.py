# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



class SelfAttn(nn.Module):
    def __init__(self, input_size):
        super(SelfAttn, self).__init__()

        self.wq = nn.Linear(input_size, input_size/2)
        self.v = nn.Linear(input_size/2, 1)

    def forward(self, question_vec, question_mask):
        """
        :param question_vec: (seq_len, batch_size, input_size)
        :param question_mask: (batch_size, seq_len)
        :return: (batch_size, input_size)
        """
        wq = self.wq(question_vec)
        wq = torch.tanh(wq)
        s = self.v(wq).squeeze(2).transpose(0, 1)  # (batch_size, seq_len)

        mask = question_mask.eq(0)
        s.masked_fill_(mask, -float('inf'))
        s = f.softmax(s, dim=1)

        result = torch.bmm(s.unsqueeze(1), question_vec.transpose(0, 1)).squeeze(1)

        return result
