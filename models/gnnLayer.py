import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules import Module
from torch_geometric.nn import GATConv

from utils import normalize_mx_adj, normalize_adj_torch


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    GCN
    input_size is the (input window size - output window size)
    hidden size is the hidden size
    num_class is the output window size
    or
    input_size is the input window size
    hidden size is the hidden size
    num_class is the input window size
    """

    def __init__(self, input_size, hidden_size, num_class, feature, hidden_layer=1, dropout=0.2, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, num_class)
        self.dropout = dropout

        self.seW = nn.Linear(feature, feature)

    def compute_adj(self, x, adj):
        # 通过点积来计算初始的相似性，然后在通过转职的矩阵转为adj，并和原始的adj之间进行加权和。
        # 另一种直接通过在原始的序列之上增加结构编码来实现更好的效果。
        src = x
        # x = x.transpose(1, 0)
        # b, w, f = x.shape
        src = src / torch.norm(src, dim=-1, keepdim=True)
        xse = torch.mm(src, src.T)
        xse = F.relu(self.seW(xse))
        adj = F.relu(xse + adj).cuda()
        adj = normalize_adj_torch(adj)
        return src, adj

    def forward(self, x, adj):
        b, w, f = x.shape
        x = x.transpose(2, 1)
        adjs = []
        for batch in range(b):
            tmp = x[batch].detach()
            tmp, adj_tmp = self.compute_adj(tmp, adj)  # TODO 设置每一步都实现adj的自动更新
            tmp = self.gcn1(tmp, adj).detach()
            x[batch] = F.relu(self.gcn2(tmp, adj)).detach()
            adjs.append(adj_tmp)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(2, 1)
        return F.log_softmax(x), adjs


class VerticalGAT(nn.Module):
    def __init__(self, d_model, feature_size, hidden_size, input_window):
        super(VerticalGAT, self).__init__()
        self.d_model = d_model
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.q_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.k_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.v_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.top_value = math.ceil(feature_size * 0.2)

        # 长期依赖短期依赖
        self.prior_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.series_projection = nn.Conv1d(d_model, d_model, 3, padding=1)

        # self._projection = nn.Conv1d(feature_size, d_model, 3, padding=1)
        self._projection = nn.Linear(d_model, d_model)

        self.gat = GAT(n_feat=input_window, n_hid=input_window // 2, out_feat=input_window, n_heads=4)

    def forward(self, x):
        b, w, f = x.shape
        x = x.transpose(2, 1)
        q, k, v, prior, series = x, x, x, x, x
        q = self.q_projection(q)
        k = self.k_projection(k)
        v = self.v_projection(v)

        scale = 1. / math.sqrt(q.shape[-2])

        adj = torch.matmul(q, k.transpose(2, 1))
        adj = adj * scale
        adj = F.sigmoid(adj)

        x = self.gat(v, adj)

        # x = self._projection(x).transpose(2, 1)
        x = self._projection(x.transpose(2, 1))
        """
        加入新的adj，gat对时间的长度进行建模，每一个数据点代表的是一个维度，
        相当于注意力作用于adj的生成部分，每个点之间的权重对比。
        同时注意，每次获得的都是最后一点对于窗口之前的权重，这样来保证不涉及数据的泄漏。
        """
        return x, adj


class HorizontalGAT(nn.Module):
    def __init__(self, d_model, feature_size, hidden_size, input_window):
        super(HorizontalGAT, self).__init__()
        self.d_model = d_model
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.input_window = input_window

        self.q_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.k_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.v_projection = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.p_projection = nn.Conv1d(d_model, d_model, 3, padding=1)

        self.gat = GAT(d_model, d_model // 2, d_model, 1)
        # self.gat = GAT(n_feat=feature_size, n_hid=feature_size//2, out_feat=feature_size, n_heads=4)
        self.distances = torch.zeros((input_window, input_window)).cuda()
        for i in range(input_window):
            for j in range(input_window):
                self.distances[i][j] = abs(i - j)

    def forward(self, x):
        b, w, f = x.shape
        x = x.transpose(2, 1)
        # shape (batch, feature, window)
        q, k, v, p = x, x, x, x
        q = self.q_projection(q)
        k = self.k_projection(k)
        v = self.v_projection(v)
        p = self.p_projection(p)
        q, k, v, p = [i.transpose(2, 1) for i in [q, k, v, p]]

        scale = 1. / math.sqrt(q.shape[-1])

        sigma = torch.matmul(p, k.transpose(2, 1))  # batch windows windows
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        prior = self.distances.unsqueeze(0).repeat(sigma.shape[0], 1, 1).cuda()  # B H L L
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        score = torch.matmul(q, k.transpose(2, 1))
        score = torch.softmax(score, dim=-1)
        score = score * scale

        x = self.gat(v, score)

        return x, score, prior


class DynamicGCN(nn.Module):
    """
    scale
    input_size is the (input window size - output window size)
    hidden size is the hidden size
    num_class is the output window size
    or
    input_size is the input window size
    hidden size is the hidden size
    num_class is the input window size
    """

    def __init__(self, input_size, hidden_size, num_class, feature, hidden_layer=1, dropout=0.2, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, num_class)
        self.dropout = dropout

        self.seW = nn.Linear(feature, feature)

    def compute_adj(self, x, adj):
        # 通过点积来计算初始的相似性，然后在通过转职的矩阵转为adj，并和原始的adj之间进行加权和。
        # 另一种直接通过在原始的序列之上增加结构编码来实现更好的效果。
        src = x
        x = x.transpose(2, 1)
        b, w, f = x.shape
        x = x.reshape(-1, f)
        xse = torch.mm(x.T, x)
        xse = F.relu(self.seW(xse))
        adj = F.relu(xse + adj)
        return src, adj

    def forward(self, x, adj):
        b, w, f = x.shape
        x = x.transpose(2, 1)
        for batch in range(b):
            tmp = x[batch].detach()
            tmp, adj_tmp = self.compute_adj(tmp, adj)  # TODO 设置每一步都实现adj的自动更新
            tmp = self.gcn1(tmp, adj_tmp).detach()
            x[batch] = F.relu(self.gcn2(tmp, adj_tmp)).detach()
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.transpose(2, 1)
        return F.log_softmax(x)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.Wlinear = nn.Linear(in_feature, out_feature)
        # self.W=nn.Parameter(torch.empty(size=(batch_size,in_feature,out_feature)))
        nn.init.xavier_uniform_(self.Wlinear.weight, gain=1.414)

        self.aiLinear = nn.Linear(out_feature, 1)
        self.ajLinear = nn.Linear(out_feature, 1)
        # self.a=nn.Parameter(torch.empty(size=(batch_size,2*out_feature,1)))
        nn.init.xavier_uniform_(self.aiLinear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight, gain=1.414)

        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def getAttentionE(self, Wh):
        # 重点改了这个函数
        Wh1 = self.aiLinear(Wh)
        Wh2 = self.ajLinear(Wh)
        Wh2 = Wh2.view(Wh2.shape[0], Wh2.shape[2], Wh2.shape[1])
        # Wh1=torch.bmm(Wh,self.a[:,:self.out_feature,:])    #Wh:size(node,out_feature),a[:out_eature,:]:size(out_feature,1) => Wh1:size(node,1)
        # Wh2=torch.bmm(Wh,self.a[:,self.out_feature:,:])    #Wh:size(node,out_feature),a[out_eature:,:]:size(out_feature,1) => Wh2:size(node,1)

        e = Wh1 + Wh2  # broadcast add, => e:size(node,node)
        return self.leakyRelu(e)

    def forward(self, h, adj):
        # print(h.shape)
        Wh = self.Wlinear(h)
        # Wh=torch.bmm(h,self.W)   #h:size(node,in_feature),W:size(in_feature,out_feature) => Wh:size(node,out_feature)
        e = self.getAttentionE(Wh)

        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_hat = torch.bmm(attention,
                          Wh)  # attention:size(node,node),Wh:size(node,out_fature) => h_hat:size(node,out_feature)

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_feature) + '->' + str(self.out_feature) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, out_feat, n_heads, dropout=0.2, alpha=0.1):
        """Dense version of GAT
        n_feat:input dim
        n_hid:hidden dim
        n_class:output dim
        alpha:leaky relu alpha
        n_heads:head size
        for time series:
        n_feat is input window size
        n_hid is hidden size
        n_class is output window size
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        # self.attentions = [GATConv(n_feat, n_hid) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        self.out_att = GraphAttentionLayer(n_hid * n_heads, out_feat, dropout=dropout, alpha=alpha, concat=False)
        # self.out_att = GATConv(n_hid * n_heads, out_feat)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
