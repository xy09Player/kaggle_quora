# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f


class Embedding(nn.Module):
    """ standard embedding """
    def __init__(self, embedding):
        super(Embedding, self).__init__()

        self.vocab_size = embedding.shape[0]
        self.w2v_size = embedding.shape[1]

        self.embedding_fix = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.w2v_size,
            padding_idx=0,
            _weight=torch.Tensor(embedding)
        )
        self.embedding_fix.weight.requires_grad = False

        self.embedding_v = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.w2v_size,
            padding_idx=0
        )

        self.embedding_dim = self.embedding_fix.embedding_dim

    def forward(self, tensor):
        """
        :param tensor: (batch_size, c_len)
        :return: (batch_size, c_len, w2v)
        """
        embedding_1 = self.embedding_fix(tensor)

        tensor = tensor - (self.vocab_size - self.embedding_v.num_embeddings)
        tensor = f.relu(tensor)
        embedding_2 = self.embedding_v(tensor)

        embedding = embedding_1 + embedding_2

        return embedding
