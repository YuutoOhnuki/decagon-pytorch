#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import scipy.sparse as sp


class Batch(object):
    def __init__(self, adjacency_matrix):
        pass

    def get(size):
        pass


def train_test_split(data, train_size=.8):
    pass


class Minibatch(object):
    def __init__(self, data, node_type_row, node_type_column, size):
        self.data = data
        self.adjacency_matrix = data.get_adjacency_matrix(node_type_row, node_type_column)
        self.size = size
        self.order = np.random.permutation(adjacency_matrix.nnz)
        self.count = 0

    def reset(self):
        self.count = 0
        self.order = np.random.permutation(adjacency_matrix.nnz)

    def __iter__(self):
        adj_mat = self.adjacency_matrix
        size = self.size
        order = np.random.permutation(adj_mat.nnz)
        for i in range(0, len(order), size):
            row = adj_mat.row[i:i + size]
            col = adj_mat.col[i:i + size]
            data = adj_mat.data[i:i + size]
            adj_mat_batch = sp.coo_matrix((data, (row, col)), shape=adj_mat.shape)
            yield adj_mat_batch
        degree = self.adjacency_matrix.sum(1)



    def __len__(self):
        pass
