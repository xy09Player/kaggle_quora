# coding = utf-8
# author = xy

import torch
from torch import nn
from torch.nn import functional as f
from model.modules import embedding


class Model(nn.Module):
    """ rnn """
    def __init__(self, param):
        super(Model, self).__init__()

        self.filter_sizes = param['filter_size']
        self.num_filters = param['num_filter']
        self.dropout_p = param['dropout_p']

        # embedding
        self.embedding = embedding.Embedding(param['embedding'])

        # encoder
        d = self.embedding.embedding_dim
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, kernel_size=[k, d]) for k in self.filter_sizes])

        # outputs
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_filters*len(self.filter_sizes), self.num_filters),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.num_filters, 1),
            nn.Sigmoid()
        )
        # dropout
        self.dropout = nn.Dropout(param['dropout_p'])

    def forward(self, batch):

        questions = batch[0]

        # embedding (batch_size, seq_len, d)
        question_vec = self.embedding(questions)

        # encoder
        question_vec = question_vec.unsqueeze(1)
        question_vec = [f.relu(conv(question_vec).squeeze(3)) for conv in self.convs]
        question_vec = [f.max_pool1d(i, i.size(2)).squeeze(2) for i in question_vec]
        question_vec = torch.cat(question_vec, dim=1)  # (batch_size, self.num_filters * len(self.filter_size))

        # outputs
        question_vec = self.dropout(question_vec)
        output = self.fc1(question_vec)
        output = self.dropout(output)
        output = self.fc2(output)  # (batch_size, 1)

        return output
