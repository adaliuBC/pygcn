import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphNormalization, NodeNormalization
import pdb

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer):
        super(GCN, self).__init__()
        self.nlayer = nlayer
        gap = (nfeat-nhid)//self.nlayer
        self.gc = []
        self.gn = []
        self.nn = []
        self.gc0 = GraphConvolution(nfeat, nhid+gap*(nlayer-1))  #
        self.gn0 = GraphNormalization(nhid+gap*(nlayer-1), nhid+gap*(nlayer-1))
        self.nn0 = NodeNormalization(nhid+gap*(nlayer-1), nhid+gap*(nlayer-1))
        self.gc.append(self.gc0)
        self.gn.append(self.gn0)
        self.nn.append(self.nn0)
        for i in range(1, self.nlayer):
            self.gc2 = GraphConvolution(nhid+gap*(nlayer-i), nhid+gap*(nlayer-i-1))
            self.gn2 = GraphNormalization(nhid+gap*(nlayer-i-1), nhid+gap*(nlayer-i-1))
            self.nn2 = NodeNormalization(nhid+gap*(nlayer-i-1), nhid+gap*(nlayer-i-1))
            self.gc.append(self.gc2)
            self.gn.append(self.gn2)
            self.nn.append(self.nn2)
        self.gcfin = GraphConvolution(nhid, nclass)
        self.gnfin = GraphNormalization(nclass, nclass)
        self.nnfin = NodeNormalization(nclass, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        for i in range(0, self.nlayer):
            x = self.gc[i](x, adj, self.gn[i], self.nn[i])
            x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.gc2(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcfin(x, adj, self.gnfin, self.nnfin)  #data_num * 7
        return F.log_softmax(x, dim=1)

