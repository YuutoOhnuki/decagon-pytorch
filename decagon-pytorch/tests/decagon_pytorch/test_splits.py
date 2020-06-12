from decagon_pytorch.data import Data
import torch
from decagon_pytorch.splits import train_val_test_split_adj_mat
import pytest


def _gen_adj_mat(n_rows, n_cols):
    res = torch.rand((n_rows, n_cols)).round()
    if n_rows == n_cols:
        res -= torch.diag(torch.diag(res))
        a, b = torch.triu_indices(n_rows, n_cols)
        res[a, b] = res.transpose(0, 1)[a, b]
    return res


def train_val_test_split_1(data, train_ratio=0.8,
    val_ratio=0.1, test_ratio=0.1):

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')

    d_train = Data()
    d_val = Data()
    d_test = Data()

    for (node_type_row, node_type_col), rels in data.relation_types.items():
        for r in rels:
            adj_train, adj_val, adj_test = train_val_test_split_adj_mat(r.adjacency_matrix)
            d_train.add_relation_type(r.name, node_type_row, node_type_col, adj_train)
            d_val.add_relation_type(r.name, node_type_row, node_type_col, adj_train + adj_val)


def train_val_test_split_2(data, train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')
    for (node_type_row, node_type_col), rels in data.relation_types.items():
        for r in rels:
            adj_mat = r.adjacency_matrix
            edges = torch.nonzero(adj_mat)
            order = torch.randperm(len(edges))
            edges = edges[order, :]
            n = round(len(edges) * train_ratio)
            edges_train = edges[:n]
            n_1 = round(len(edges) * (train_ratio + val_ratio))
            edges_val = edges[n:n_1]
            edges_test = edges[n_1:]
            if len(edges_train) * len(edges_val) * len(edges_test) == 0:
                raise ValueError('Not enough edges to split into train/val/test sets for: ' + r.name)


def test_train_val_test_split_adj_mat():
    adj_mat = _gen_adj_mat(50, 100)
    adj_mat_train, adj_mat_val, adj_mat_test = \
        train_val_test_split_adj_mat(adj_mat, train_ratio=0.8,
            val_ratio=0.1, test_ratio=0.1)

    assert adj_mat.shape == adj_mat_train.shape == \
        adj_mat_val.shape == adj_mat_test.shape

    edges_train = torch.nonzero(adj_mat_train)
    edges_val = torch.nonzero(adj_mat_val)
    edges_test = torch.nonzero(adj_mat_test)

    edges_train = set(map(tuple, edges_train.tolist()))
    edges_val = set(map(tuple, edges_val.tolist()))
    edges_test = set(map(tuple, edges_test.tolist()))

    assert edges_train.intersection(edges_val) == set()
    assert edges_train.intersection(edges_test) == set()
    assert edges_test.intersection(edges_val) == set()

    assert torch.all(adj_mat_train + adj_mat_val + adj_mat_test == adj_mat)

    # assert torch.all((edges_train != edges_val).sum(1).to(torch.bool))
    # assert torch.all((edges_train != edges_test).sum(1).to(torch.bool))
    # assert torch.all((edges_val != edges_test).sum(1).to(torch.bool))


@pytest.mark.skip
def test_splits_01():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    d.add_relation_type('Interaction', 0, 0,
        _gen_adj_mat(1000, 1000))
    d.add_relation_type('Target', 1, 0,
        _gen_adj_mat(100, 1000))
    d.add_relation_type('Side Effect: Insomnia', 1, 1,
        _gen_adj_mat(100, 100))
    d.add_relation_type('Side Effect: Incontinence', 1, 1,
        _gen_adj_mat(100, 100))
    d.add_relation_type('Side Effect: Abdominal pain', 1, 1,
        _gen_adj_mat(100, 100))

    d_train, d_val, d_test = train_val_test_split(d, 0.8, 0.1, 0.1)
