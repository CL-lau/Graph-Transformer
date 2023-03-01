import numpy as np
from torch.nn import LayerNorm

from models import base_model
from models.base_model import _get_clone, TransformerEncoder, TransformerEncoderLayer, ContextConv
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import argparse
from models.embed import DataEmbedding
from models.gnnLayer import VerticalGAT


class Transformer(nn.Module):
    def __init__(self, feature_size=25, d_model=256, num_layers=3, dropout=0.1, input_window=50, adj=None,
                 batch_size=16, num_head=5):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding = DataEmbedding(feature_size, d_model, input_window)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_head, dropout=dropout,
                                                     batch_first=True, input_window=input_window, adj=adj,
                                                     batch_size=batch_size, feature_size=feature_size)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers,
                                                      d_model=d_model, batch_size=batch_size,
                                                      input_window=input_window)
        # self.decoder = nn.Sequential(
        #     ContextConv(in_channels=d_model, out_channels=d_model, k=3), nn.Linear(d_model, feature_size))
        self.decoder1 = nn.Linear(d_model, feature_size)
        self.verticalGAT = VerticalGAT(d_model=feature_size, feature_size=feature_size, hidden_size=d_model,
                                       input_window=input_window)
        self.decoder2 = nn.Linear(feature_size, feature_size)
        self.norm = LayerNorm(feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.cuda()
        src = self.embedding(src)
        output, atte_s, adj_s, series_s, prior_s = self.transformer_encoder(src)

        # TODO FULL
        output = self.decoder1(output)
        out = output.clone()
        tmp, adj = self.verticalGAT(output)
        output = self.norm(output + tmp)
        output = self.decoder2(output)

        # # # TODO GRAPH
        # output = self.decoder1(output)
        # out = output.clone()
        # adj = torch.randn((out.shape[-1], out.shape[-1])).cuda()

        return output, atte_s, adj, series_s, prior_s, out
