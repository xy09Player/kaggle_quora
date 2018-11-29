# coding = utf-8
# author = xy


import torch
from torch import nn


class Rnn(nn.Module):

    def __init__(self, param):
        super(Rnn, self).__init__()

        self.mode = param['mode']
        self.input_size = param['input_size']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['encoder_dropout_p']
        self.directional = True
        self.layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        if self.mode == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.layer_num,
                bidirectional=self.directional,
                dropout=self.dropout_p if self.layer_num > 1 else 0
            )
        elif self.mode == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.layer_num,
                bidirectional=self.directional,
                dropout=self.dropout_p if self.layer_num > 1 else 0
            )

        if self.is_bn:
            self.layer_norm = nn.LayerNorm(self.input_size)

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        """ use xavier_uniform to initialize rnn weights """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, vec, mask):
        """
        :param vec: (seq_len, batch_size, input_size)
        :param mask: (batch_size, seq_len)
        :return: (seq_len, batch_size, hidden_size*directional_num)
        """

        # layer normalization
        if self.is_bn:
            seq_len, batch_size, input_size = vec.size
            vec = vec.contiguous().view(-1, input_size)
            vec = self.layer_norm(vec)
            vec = vec.view(seq_len, batch_size, input_size)

        # dropout
        vec = self.dropout(vec)

        # forward
        lengths = mask.long().sum(1)
        length_sort, idx_sort = torch.sort(lengths, descending=True)
        _, idx_unsort = torch.sort(idx_sort)

        v_sort = vec.index_select(1, idx_sort)
        v_pack = nn.utils.rnn.pack_padded_sequence(v_sort, length_sort)
        outputs, _ = self.rnn(v_pack, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.index_select(1, idx_unsort)

        # 未填充， outputs的第一维可能小于seq_len
        return outputs
