import math
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
from torch.nn.utils import weight_norm

import pdb

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
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

    def forward(self, input, adj, gn_func, nn_func):
        #pdb.set_trace()
        support = torch.mm(input, self.weight)
        
        print(support.shape, self.weight.shape, input.shape)
        #support = nn_func(support)
        
        output = F.relu(support)
        print(adj.shape, output.shape)
        output = torch.spmm(adj, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphNormalization(Module):

    def __init__(self, in_size, out_size):
        super(GraphNormalization, self).__init__()

        self.trans = weight_norm(nn.Linear(in_size, out_size))


    def forward(self, input):

        out_mean = torch.mean(torch.mean(input, 1, keepdim=True), 0, keepdim=True).expand_as(input)  
        #计算行和列所有元素的总平均值，形成一个由mean值构成的、和input形状相同的矩阵
        out_var = torch.sum(torch.sum((input - out_mean)*(input - out_mean), 1, keepdim=True), 0, keepdim=True)
        #求所有项值和avg值的差之和
        out_var = out_var/(input.shape[0]*input.shape[1])  #求方差（比上一步除以矩阵总元素数，即求avg）
        out_var = torch.sqrt(out_var).expand_as(input)  #求标准差，即逐元素求sqrt
        out = (input-out_mean)/(out_var+1.0e-10)  #减均值除标准差
        out = self.trans(out)

        return out


class NodeNormalization(Module):
    #对每个node进行normalization
    def __init__(self, in_size, out_size):
        super(NodeNormalization, self).__init__()

        self.trans = weight_norm(nn.Linear(in_size, out_size))

    def forward(self, input):

        out_mean = torch.mean(input, 1, keepdim=True).expand_as(input)  #对每行求平均
        out_var = torch.mean((input - out_mean+1.0e-10)*(input - out_mean+1.0e-10), 1, keepdim=True)  #对每行求方差
        out_var = torch.sqrt(out_var+1.0e-10).expand_as(input)  #求标准差
        out = (input-out_mean)/(out_var+1.0e-10)  #减均值除标准差
        out = self.trans(out)

        return out

#如果是对每个feature进行normalization呢？少出现的feature有更多意义……？
