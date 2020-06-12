#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from icosagon.trainprep import TrainValTest, \
    train_val_test_split_edges, \
    get_edges_and_degrees, \
    prepare_adj_mat, \
    prepare_relation_type
import torch
import pytest
import numpy as np
from itertools import chain
from icosagon.data import RelationType


def test_train_val_test_split_edges_01():
    edges = torch.randint(0, 10, (10, 2))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(edges, TrainValTest(.5, .5, .5))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(edges, TrainValTest(.2, .2, .2))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(edges, None)
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(edges, (.8, .1, .1))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(np.random.randint(0, 10, (10, 2)), TrainValTest(.8, .1, .1))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(torch.randint(0, 10, (10, 3)), TrainValTest(.8, .1, .1))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(torch.randint(0, 10, (10, 2, 1)), TrainValTest(.8, .1, .1))
    with pytest.raises(ValueError):
        _ = train_val_test_split_edges(None, TrainValTest(.8, .2, .2))
    res = train_val_test_split_edges(edges, TrainValTest(.8, .1, .1))
    assert res.train.shape == (8, 2) and res.val.shape == (1, 2) and \
        res.test.shape == (1, 2)
    res = train_val_test_split_edges(edges, TrainValTest(.8, .0, .2))
    assert res.train.shape == (8, 2) and res.val.shape == (0, 2) and \
        res.test.shape == (2, 2)
    res = train_val_test_split_edges(edges, TrainValTest(.8, .2, .0))
    assert res.train.shape == (8, 2) and res.val.shape == (2, 2) and \
        res.test.shape == (0, 2)
    res = train_val_test_split_edges(edges, TrainValTest(.0, .5, .5))
    assert res.train.shape == (0, 2) and res.val.shape == (5, 2) and \
        res.test.shape == (5, 2)
    res = train_val_test_split_edges(edges, TrainValTest(.0, .0, 1.))
    assert res.train.shape == (0, 2) and res.val.shape == (0, 2) and \
        res.test.shape == (10, 2)
    res = train_val_test_split_edges(edges, TrainValTest(.0, 1., .0))
    assert res.train.shape == (0, 2) and res.val.shape == (10, 2) and \
        res.test.shape == (0, 2)


def test_train_val_test_split_edges_02():
    edges = torch.randint(0, 30, (30, 2))
    ratios = TrainValTest(.8, .1, .1)
    res = train_val_test_split_edges(edges, ratios)
    edges = [ tuple(a) for a in edges ]
    res = [ tuple(a) for a in chain(res.train, res.val, res.test) ]
    assert all([ a in edges for a in res ])


def test_get_edges_and_degrees_01():
    adj_mat_dense = (torch.rand((10, 10)) > .5)
    adj_mat_sparse = adj_mat_dense.to_sparse()
    edges_dense, degrees_dense = get_edges_and_degrees(adj_mat_dense)
    edges_sparse, degrees_sparse = get_edges_and_degrees(adj_mat_sparse)
    assert torch.all(degrees_dense == degrees_sparse)
    edges_dense = [ tuple(a) for a in edges_dense ]
    edges_sparse = [ tuple(a) for a in edges_dense ]
    assert len(edges_dense) == len(edges_sparse)
    assert all([ a in edges_dense for a in edges_sparse ])
    assert all([ a in edges_sparse for a in edges_dense ])
    # assert torch.all(edges_dense == edges_sparse)


def test_prepare_adj_mat_01():
    adj_mat = (torch.rand((10, 10)) > .5)
    adj_mat = adj_mat.to_sparse()
    ratios = TrainValTest(.8, .1, .1)
    _ = prepare_adj_mat(adj_mat, ratios)


def test_prepare_adj_mat_02():
    adj_mat = (torch.rand((10, 10)) > .5)
    adj_mat = adj_mat.to_sparse()
    ratios = TrainValTest(.8, .1, .1)
    (adj_mat_train, edges_pos, edges_neg) = prepare_adj_mat(adj_mat, ratios)
    assert isinstance(adj_mat_train, torch.Tensor)
    assert adj_mat_train.is_sparse
    assert adj_mat_train.shape == adj_mat.shape
    assert adj_mat_train.dtype == adj_mat.dtype
    assert isinstance(edges_pos, TrainValTest)
    assert isinstance(edges_neg, TrainValTest)
    for a in ['train', 'val', 'test']:
        for b in [edges_pos, edges_neg]:
            edges = getattr(b, a)
            assert isinstance(edges, torch.Tensor)
            assert len(edges.shape) == 2
            assert edges.shape[1] == 2


def test_prepare_relation_type_01():
    adj_mat = (torch.rand((10, 10)) > .5)
    r = RelationType('Test', 0, 0, adj_mat, True)
    ratios = TrainValTest(.8, .1, .1)
    _ = prepare_relation_type(r, ratios, False)



# def prepare_relation(r, ratios):
#     adj_mat = r.adjacency_matrix
#     adj_mat_train, edges_pos, edges_neg = prepare_adj_mat(adj_mat)
#
#     if r.node_type_row == r.node_type_column:
#         adj_mat_train = norm_adj_mat_one_node_type(adj_mat_train)
#     else:
#         adj_mat_train = norm_adj_mat_two_node_types(adj_mat_train)
#
#     return PreparedRelation(r.name, r.node_type_row, r.node_type_column,
#         adj_mat_train, edges_pos, edges_neg)
