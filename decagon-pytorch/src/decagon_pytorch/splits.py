import torch
from .data import Data, \
    AdjListData


def train_val_test_split_adj_mat(adj_mat, train_ratio, val_ratio, test_ratio,
    return_edges=False):

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')

    edges = torch.nonzero(adj_mat)
    order = torch.randperm(len(edges))
    edges = edges[order, :]
    n = round(len(edges) * train_ratio)
    edges_train = edges[:n]
    n_1 = round(len(edges) * (train_ratio + val_ratio))
    edges_val = edges[n:n_1]
    edges_test = edges[n_1:]

    adj_mat_train = torch.zeros_like(adj_mat)
    adj_mat_train[edges_train[:, 0], edges_train[:, 1]] = 1

    adj_mat_val = torch.zeros_like(adj_mat)
    adj_mat_val[edges_val[:, 0], edges_val[:, 1]] = 1

    adj_mat_test = torch.zeros_like(adj_mat)
    adj_mat_test[edges_test[:, 0], edges_test[:, 1]] = 1

    res = (adj_mat_train, adj_mat_val, adj_mat_test)
    if return_edges:
        res += (edges_train, edges_val, edges_test)

    return res


def train_val_test_split_edges(adj_mat, train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')

    edges = torch.nonzero(adj_mat)
    order = torch.randperm(len(edges))
    edges = edges[order, :]
    n = round(len(edges) * train_ratio)
    edges_train = edges[:n]
    n_1 = round(len(edges) * (train_ratio + val_ratio))
    edges_val = edges[n:n_1]
    edges_test = edges[n_1:]

    adj_mat_train = torch.zeros_like(adj_mat)
    adj_mat_train[edges_train[:, 0], edges_train[:, 1]] = 1

    adj_mat_val = torch.zeros_like(adj_mat)
    adj_mat_val[edges_val[:, 0], edges_val[:, 1]] = 1

    adj_mat_test = torch.zeros_like(adj_mat)
    adj_mat_test[edges_test[:, 0], edges_test[:, 1]] = 1

    return adj_mat_train, adj_mat_val, adj_mat_test, \
        edges_train, edges_val, edges_test
