import copy
import math
from typing import Optional, Any, Union, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Module
# from torch.nn.modules.activation import MultiheadAttention
# from torch.nn.modules.activation import SELU
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear, NonDynamicallyQuantizableLinear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules import activation

from models.gnnLayer import VerticalGAT, HorizontalGAT
from utils import normalize_mx_adj, normalize_adj_torch


def _get_clone(model, N):
    return ModuleList([copy.deepcopy(model) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']
    """
    d_model is the feature size
    dim_feedforward is the hidden size if linear for FFN
    nhead is the head num
    input window is the length of feature
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.2, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device="cuda", feature_size=25, input_window=50, batch_size=16, adj=None):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dim_feedforward = dim_feedforward
        self.input_window = input_window
        self.d_model = d_model
        self.device = device
        self.adj = adj

        self.norm_first = norm_first
        self.activation = activation

        # # FFCN模块
        # # 卷积的通道设置为1, 其余的全部变成batch_size, FFN改为一维卷积 + GELU_BN + 三维卷积 + 残差 + GELU_BN + 一维卷积 + BN
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.con1_1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.con1_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.multiStepConv = MultiScaleConv(input_window=input_window, output_window=input_window, feature=d_model)

        self.horizontalGAT = HorizontalGAT(d_model, feature_size=25, hidden_size=d_model, input_window=input_window)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x, atte = self._sa_block(x)
        x = self.norm1(src + x)
        adj = None

        # TODO FULL
        tmp, series, prior = self.horizontalGAT(x)
        x = self.norm2(x + tmp)

        # # TODO GRAPH
        # series = None
        # prior = None

        x = self.norm3(x + self._conv1_block(x))
        return x, atte, adj, series, prior

    def _sa_block(self, x):
        x, atte = self.self_attn(x, x, x,
                                 need_weights=True)
        return self.dropout1(x), atte

    def _conv1_block(self, x):
        src = x
        src = src.transpose(2, 1)
        src = self.multiStepConv(src)
        src = src.transpose(2, 1)
        src = self.gelu2(src)
        return src


class TransformerEncoder(Module):
    __constants__ = ['norm']
    """
    d_model is the feature size 
    """

    def __init__(self, encoder_layer, num_layers, d_model, batch_size, input_window, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clone(encoder_layer, num_layers)
        self.num_layer = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        x = src.cuda()
        i = 0
        atte_s = []
        adj_s = []
        series_s = []
        prior_s = []
        for mod in self.layers:
            x, atte, adj, series, prior = mod(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            atte_s.append(atte)
            adj_s.append(adj)
            series_s.append(series)
            prior_s.append(prior)
            i = i+1
        if self.norm is not None:
            x = self.norm(x)
        return x, atte_s, adj_s, series_s, prior_s


class MultiheadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim=512, num_heads=8, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query_projection = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.key_projection = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.value_projection = nn.Linear(embed_dim, num_heads * self.head_dim)

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, padding=0)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))


class ContextConv(torch.nn.Module):
    """"
    causal_convolution_layer parameters:
    in_channels: the number of features per time point
    out_channels: the number of features outputted per time point
    kernel_size: k is the width of the 1-D sliding kernel
    """

    def __init__(self, in_channels=25, out_channels=25, k=5, batch_first=True):
        super(ContextConv, self).__init__()
        self.batch_first = batch_first
        self.causal_convolution = CausalConv1d(in_channels, out_channels, kernel_size=k)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, window, feature)->(batch, feature, window)
        x = self.causal_convolution(x)
        x = x.transpose(1, 2)  # (batch, feature, window)->(batch, window, feature)
        return F.tanh(x)


# RNN的两种实现方式:一个窗口一个窗口的递归，另一种一步一步的实现递归。
class ComputeREC(nn.Module):
    """
    in_feature is feature num
    hidden size is hidden num
    n_class is the output feature num
    """

    def __init__(self, in_feature, hidden_size, n_class):
        super(ComputeREC, self).__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.in2hidden = nn.Linear(in_feature + self.hidden_size, self.hidden_size)
        self.hidden2out = nn.Linear(self.hidden_size, self.n_class)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.pre_state = None

    # input shape is [batch,seq_len,in_feature]
    def forward(self, input, pre_state):
        batch, w, f = input.shape
        a = torch.zeros(batch, w, self.hidden_size, requires_grad=False).cuda()  # a-> [T,hidden_size]
        out = torch.zeros(batch, w, self.n_class, requires_grad=False).cuda()
        self.pre_state = pre_state

        if self.pre_state is None:
            self.pre_state = torch.zeros(w, self.hidden_size,
                                         requires_grad=False).cuda()  # hidden ->[batch,hidden_size]

        for b in range(batch):
            tmp = torch.cat((input[b], self.pre_state), 1)  # [w,in_feature]+[w,hidden_size]-> [w,hidden_size+in_featue]
            a[b] = self.in2hidden(
                tmp)  # [w,hidden_size+in_feature]*[hidden_size+in_feature,hidden_size] ->[w,hidden_size]
            hidden = self.tanh(a[b])

            self.pre_state = hidden.detach()
            out[b] = self.softmax(self.hidden2out(hidden))  # [w,hidden_size]*[hidden_size,n_class]->[w,n_class]
        return out, self.pre_state


class MultiScaleConv(nn.Module):
    def __init__(self, input_window, output_window, feature):
        super(MultiScaleConv, self).__init__()
        assert input_window == output_window
        self.conv3 = nn.Conv1d(in_channels=feature, out_channels=feature, padding=1, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=feature, out_channels=feature, padding=2, kernel_size=5)
        self.conv7 = nn.Conv1d(in_channels=feature, out_channels=feature, padding=3, kernel_size=7)
        self.linear = nn.Sequential(nn.Linear(in_features=3 * input_window, out_features=2 * input_window),
                                    nn.Linear(in_features=2 * input_window, out_features=output_window))

    def forward(self, x):
        """
            the shape of x is (batch, feature, input_window)
            the output shape is (batch, feature , input_window)
        """
        tmp3 = self.conv3(x)
        sig3 = F.sigmoid(tmp3)
        tan3 = F.tanh(tmp3)
        tmp3 = sig3 * tan3
        tmp5 = self.conv5(x)
        sig5 = F.sigmoid(tmp5)
        tan5 = F.tanh(tmp5)
        tmp5 = sig5 * tan5
        tmp7 = self.conv7(x)
        sig7 = F.sigmoid(tmp7)
        tan7 = F.tanh(tmp7)
        tmp7 = sig7 * tan7
        # tmp3, tmp5, tmp7 shape is (batch, feature, input_window)
        # tmp shape is (batch, feature, 3*input_window) -> (batch, feature, input_window)
        tmp = torch.cat((tmp3, tmp5, tmp7), dim=2)
        x = self.linear(tmp)
        x = F.sigmoid(x)
        return x
