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
        support = torch.mm(input, self.weight)
        
        support = nn_func(support)
        
        output = F.relu(support)
        
        #output = nn_func(output)
        
        output = torch.spmm(adj, support)
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
        #�����к�������Ԫ�ص���ƽ��ֵ���γ�һ����meanֵ���ɵġ���input��״��ͬ�ľ���
        out_var = torch.sum(torch.sum((input - out_mean)*(input - out_mean), 1, keepdim=True), 0, keepdim=True)
        #��������ֵ��avgֵ�Ĳ�֮��
        out_var = out_var/(input.shape[0]*input.shape[1])  #�󷽲����һ�����Ծ�����Ԫ����������avg��
        out_var = torch.sqrt(out_var).expand_as(input)  #���׼�����Ԫ����sqrt
        out = (input-out_mean)/(out_var+1.0e-10)  #����ֵ����׼��
        out = self.trans(out)

        return out


class NodeNormalization(Module):
    #��ÿ��node����normalization
    def __init__(self, in_size, out_size):
        super(NodeNormalization, self).__init__()

        self.trans = weight_norm(nn.Linear(in_size, out_size))
        
        self.fc1_m = nn.Linear(in_size, (1+in_size)//2)
        self.relu1_m = nn.ReLU(True)
        self.fc2_m = nn.Linear((1+in_size)//2, 1)
        self.relu2_m = nn.ReLU(True)
        self.net_mean = nn.Sequential(self.fc1_m, self.relu1_m, self.fc2_m, self.relu2_m)
        
        self.fc1_v = nn.Linear(in_size+1, (1+in_size)//2)
        self.relu1_v = nn.ReLU(True)
        self.fc2_v = nn.Linear((1+in_size)//2, 1)
        self.relu2_v = nn.ReLU(True)
        self.net_var = nn.Sequential(self.fc1_v, self.relu1_v, self.fc2_v, self.relu2_v)
        
        #self.beta_net = nn.Conv1d(in_size, 1, 1)
        #self.gamma_net = nn.Conv1d(in_size, 1, 1)
        self.train_beta = torch.nn.Parameter(torch.FloatTensor(1))
        self.train_gamma = torch.nn.Parameter(torch.FloatTensor(1))
        

    def forward(self, input):
        #comp_mean, comp_var
        #out_mean = torch.mean(input, 1, keepdim=True).expand_as(input)  #��ÿ����ƽ��
        #out_var = torch.mean((input - out_mean+1.0e-10)*(input - out_mean+1.0e-10), 1, keepdim=True)  #��ÿ���󷽲�
        #out_var = torch.sqrt(out_var+1.0e-10).expand_as(input)  #���׼��
        #out = (input-out_mean)/(out_var+1.0e-10)  #����ֵ����׼��
        #out = self.trans(out)

        #train_mean, comp_var
        row_num = (input.size())[0]
        train_mean = self.net_mean(input)
        train_mean = train_mean.expand(row_num, 1)
        cat = torch.cat((input, train_mean), 1)
        train_var = self.net_var(cat)
        comp_mean = torch.mean(input, 1, keepdim=True).expand_as(input)  #��ÿ����ƽ��
        comp_var = torch.mean((input - comp_mean+1.0e-10)*(input - comp_mean+1.0e-10), 1, keepdim=True)  #��ÿ���󷽲�
        comp_var = torch.sqrt(comp_var+1.0e-10).expand_as(input)  #���׼��
        out = (input-train_mean)/(comp_var+1.0e-10)
        
        #train_beta, gamma
        #comp_mean = torch.mean(input, 1, keepdim=True).expand_as(input)  #��ÿ����ƽ��
        #comp_var = torch.mean((input - comp_mean+1.0e-10)*(input - comp_mean+1.0e-10), 1, keepdim=True)  #��ÿ���󷽲�
        #comp_var = torch.sqrt(comp_var+1.0e-10).expand_as(input)  #���׼��
        #comp_in = (input-comp_mean)/(comp_var+1.0e-10)
        #print(self.train_gamma, self.train_beta)
        #row_num = (input.size())[0]
        #train_beta = torch.Tensor(self.train_beta)
        #train_beta = train_beta.expand(row_num, 1)
        #out = comp_in*self.train_gamma + train_beta
        
        return out

#����Ƕ�ÿ��feature����normalization�أ��ٳ��ֵ�feature�и������塭����
