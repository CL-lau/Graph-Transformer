import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

from models.base_model import ContextConv


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, input_window):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.contextConv = ContextConv(in_channels=c_in, out_channels=c_in, k=3)
        self.contextConv1 = ContextConv(in_channels=c_in, out_channels=c_in, k=1)
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.seW = nn.Linear(c_in, input_window)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def structure_embedding(self, x):
        b, w, f = x.shape
        src = x.detach()
        xse = torch.empty(b, f, f).cuda()
        for batch in range(b):
            # compute the cos similarity of x
            src[batch] = src[batch] / torch.norm(src[batch], dim=-1, keepdim=True)
            xse[batch] = torch.mm(src[batch].T, src[batch])
        xse = F.relu(self.seW(xse))
        xse = xse.transpose(2, 1)
        return xse

    def forward(self, x):
        src = x
        x = self.contextConv(x)
        # x = x + self.contextConv1(x)
        x = x + self.structure_embedding(src)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, input_window, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, input_window=input_window)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
