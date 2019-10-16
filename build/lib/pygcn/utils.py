import numpy as np
import scipy.sparse as sp
import torch
import random

import pdb

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #将txt每行存进(行数×列数)的列表里
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = np.array(features.todense())
    labels = encode_onehot(idx_features_labels[:, -1])  #one_hot编码class
    ##
    print(type(features))
    print(labels)
    ##
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  #idx--paper_id
    idx_map = {j: i for i, j in enumerate(idx)}  #paper_id:idx
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)  #读取edge关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape) #对edges中的每一个，在idx_map找到对应项
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  #.multiply是逐元素相乘
    print(adj[2698,2697], adj[0, 0])
    features = normalize(features)   #把features的每列从(0,0,1,0,1,0)变成(0,0,1/2,0,1/2,0)即 使和为1
    adj = normalize(adj + sp.eye(adj.shape[0]))  #eye:对角线为1的稀疏矩阵
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    
    i = 0
    j = 0
    feat_index=2708
    print(adj.shape)
    print(adj[0, 9])
    for i in range(0, 2708):
        for j in range(i+1, 2708):
            #print("--sta--")
            if(adj[i, j]!=0):  #if这两点之间有edge
                print(i, j)
                print(adj[i, j])
                lambd = random.uniform(0, 1)
                #+feature
                feature_edge = lambd*features[i] + (1-lambd)*features[j]
                feature_edge = feature_edge[:, np.newaxis]
                #+label
                #label_edge = lambd*labels[i] + (1-lambd)*labels[j]
                label_fir = lambd*labels[i]
                label_sec = (1-lambd)*labels[j]
                #label_edge = label_edge[:, np.newaxis]
                label_fir = label_fir[:, np.newaxis]
                label_sec = label_sec[:, np.newaxis]
                #+adj
                #adj_edge = lambd*adj[:, i] + (1-lambd)*adj[:, j]
                #adj_edge = sp.coo_matrix((2708, 1), dtype=np.float32)
                adj_edge = lambd*adj[:, i] + (1-lambd)*adj[:, j]
                #adj_edge = adj_edge[np.newaxis, :]
                features = np.r_[features, feature_edge.T]
                features = np.r_[features, feature_edge.T]
                labels = np.r_[labels, label_fir.T]
                labels = np.r_[labels, label_sec.T]
                print("1:", adj.shape, adj_edge.shape)
                print(type(adj))
                #adj_edge = adj_edge.reshape((2708, 1))
                #adj = np.insert(adj, 2708+feat_index, values=adj_edge, axis=1)
                #adj=np.column_stack((adj,adj_edge))
                adj = sp.hstack([adj, adj_edge]).tocsr()
                adj = sp.hstack([adj, adj_edge]).tocsr()
                print("2:", adj.shape, adj_edge.shape)
                feat_index += 1
            #print(adj.shape)
            #print("hello")
        print("--end--")
        print(features.shape)
        print(labels.shape)
        print(adj.shape)
        print(feat_index)
    idx_edge = range(2708, feat_index)
    
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])
    #labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_edge = torch.LongTensor(idx_edge)
    
    return adj, features, labels, idx_train, idx_val, idx_test #, idx_edge


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #把矩阵每行求和，压缩成一列
    r_inv = np.power(rowsum, -1).flatten()  #power：对arg1中的每个元素求arg2次方
    r_inv[np.isinf(r_inv)] = 0.    #把那些不小心被置为无穷的值恢复成0
    r_mat_inv = sp.diags(r_inv) #创造一个对角矩阵
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
