# coding = utf-8
# author = xy

import torch
from torch import nn
from model.modules import embedding
from model.modules import encoder


class Model(nn.Module):
    """ rnn """
    def __init__(self, param):
        super(Model, self).__init__()

        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']

        # embedding
        self.embedding = embedding.Embedding(param['embedding'])

        # encoder
        param['input_size'] = self.embedding.embedding_dim
        self.encoder = encoder.Rnn(param)

        # outputs
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # dropout
        self.dropout = nn.Dropout(param['dropout_p'])

    def forward(self, batch):

        questions = batch[0]

        # mask
        def get_mask(tensor): return torch.ne(tensor, 0)
        question_mask = get_mask(questions)

        # embedding
        question_vec = self.embedding(questions)
        question_vec = question_vec.transpose(0, 1)

        # encoder (seq_len, batch_size, h*2)
        question_vec = self.encoder(question_vec, question_mask)

        # output
        question_vec = torch.sum(question_vec, dim=0)
        question_mask = question_mask.long().sum(1)
        question_mask = question_mask.view(-1, 1).float()
        question_vec = question_vec / question_mask  # (batch_size, h*2)

        question_vec = self.dropout(question_vec)
        output = self.fc1(question_vec)
        output = self.dropout(output)
        output = self.fc2(output)  # (batch_size, 1)

        return output
