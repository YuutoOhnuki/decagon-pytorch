#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def normalize_adjacency_matrix(adj):
    adj = sp.coo_matrix(adj)

    if adj.shape[0] == adj.shape[1]:
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = np.power(rowsum, -0.5).flatten()
        degree_mat_inv_sqrt = sp.diags(degree_mat_inv_sqrt)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    else:
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
    return sparse_to_tuple(adj_normalized)


def norm_adj_mat_one_node_type(adj):
    adj = sp.coo_matrix(adj)
    assert adj.shape[0] == adj.shape[1]
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.power(rowsum, -0.5).flatten()
    degree_mat_inv_sqrt = sp.diags(degree_mat_inv_sqrt)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized


def norm_adj_mat_two_node_types(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    colsum = np.array(adj.sum(0))
    rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
    coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
    adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
    return adj_normalized
