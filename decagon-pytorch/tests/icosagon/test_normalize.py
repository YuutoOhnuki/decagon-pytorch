from icosagon.normalize import add_eye_sparse, \
    norm_adj_mat_one_node_type_sparse, \
    norm_adj_mat_one_node_type_dense, \
    norm_adj_mat_one_node_type, \
    norm_adj_mat_two_node_types_sparse, \
    norm_adj_mat_two_node_types_dense, \
    norm_adj_mat_two_node_types
import decagon_pytorch.normalize
import torch
import pytest
import numpy as np
from math import sqrt


def test_add_eye_sparse_01():
    adj_mat_dense = torch.rand((10, 10))
    adj_mat_sparse = adj_mat_dense.to_sparse()

    adj_mat_dense += torch.eye(10)
    adj_mat_sparse = add_eye_sparse(adj_mat_sparse)

    assert torch.all(adj_mat_sparse.to_dense() == adj_mat_dense)


def test_add_eye_sparse_02():
    adj_mat_dense = torch.rand((10, 20))
    adj_mat_sparse = adj_mat_dense.to_sparse()

    with pytest.raises(ValueError):
        _ = add_eye_sparse(adj_mat_sparse)


def test_add_eye_sparse_03():
    adj_mat_dense = torch.rand((10, 10))

    with pytest.raises(ValueError):
        _ = add_eye_sparse(adj_mat_dense)


def test_add_eye_sparse_04():
    adj_mat_dense = np.random.rand(10, 10)

    with pytest.raises(ValueError):
        _ = add_eye_sparse(adj_mat_dense)


def test_norm_adj_mat_one_node_type_sparse_01():
    adj_mat = torch.rand((10, 10))
    adj_mat = (adj_mat > .5)
    adj_mat = adj_mat.to_sparse()
    _ = norm_adj_mat_one_node_type_sparse(adj_mat)


def test_norm_adj_mat_one_node_type_sparse_02():
    adj_mat_dense = torch.rand((10, 10))
    adj_mat_dense = (adj_mat_dense > .5)
    adj_mat_sparse = adj_mat_dense.to_sparse()
    adj_mat_sparse = norm_adj_mat_one_node_type_sparse(adj_mat_sparse)
    adj_mat_dense = norm_adj_mat_one_node_type_dense(adj_mat_dense)
    assert torch.all(adj_mat_sparse.to_dense() - adj_mat_dense < 0.000001)


def test_norm_adj_mat_one_node_type_dense_01():
    adj_mat = torch.rand((10, 10))
    adj_mat = (adj_mat > .5)
    _ = norm_adj_mat_one_node_type_dense(adj_mat)


def test_norm_adj_mat_one_node_type_dense_02():
    adj_mat = torch.tensor([
        [0, 1, 1, 0], # 3
        [1, 0, 1, 0], # 3
        [1, 1, 0, 1], # 4
        [0, 0, 1, 0]  # 2
    #    3  3  4  2
    ])
    expect_denom = np.array([
        [ 3,           3,           sqrt(3)*2,   sqrt(6)   ],
        [ 3,           3,           sqrt(3)*2,   sqrt(6)   ],
        [ sqrt(3)*2,   sqrt(3)*2,   4,           sqrt(2)*2 ],
        [ sqrt(6),     sqrt(6),     sqrt(2)*2,   2         ]
    ], dtype=np.float32)
    expect = (adj_mat.detach().cpu().numpy().astype(np.float32) + np.eye(4)) / expect_denom
    # expect = np.array([
    #     [1/3, 1/3, 1/3, 0],
    #     [1/3, 1/3, 1/3, 0],
    #     [1/4, 1/4, 1/4, 1/4],
    #     [0, 0, 1/2, 1/2]
    # ], dtype=np.float32)
    res = decagon_pytorch.normalize.norm_adj_mat_one_node_type(adj_mat)
    res = res.todense().astype(np.float32)
    print('res:', res)
    print('expect:', expect)
    assert np.all(res - expect < 0.000001)


def test_norm_adj_mat_one_node_type_dense_03():
    # adj_mat = torch.rand((10, 10))
    adj_mat = torch.tensor([
        [0, 1, 1,  0,  0],
        [1, 0, 1,  0,  1],
        [1, 1, 0, .5, .5],
        [0, 0, .5, 0,  1],
        [0, 1, .5, 1,  0]
    ])
    # adj_mat = (adj_mat > .5)
    adj_mat_dec = decagon_pytorch.normalize.norm_adj_mat_one_node_type(adj_mat)
    adj_mat_ico = norm_adj_mat_one_node_type_dense(adj_mat)
    adj_mat_dec = adj_mat_dec.todense()
    adj_mat_ico = adj_mat_ico.detach().cpu().numpy()
    print('adj_mat_dec:', adj_mat_dec)
    print('adj_mat_ico:', adj_mat_ico)
    assert np.all(adj_mat_dec - adj_mat_ico < 0.000001)


def test_norm_adj_mat_two_node_types_sparse_01():
    adj_mat = torch.rand((10, 20))
    adj_mat = (adj_mat > .5)
    adj_mat = adj_mat.to_sparse()
    _ = norm_adj_mat_two_node_types_sparse(adj_mat)


def test_norm_adj_mat_two_node_types_sparse_02():
    adj_mat_dense = torch.rand((10, 20))
    adj_mat_dense = (adj_mat_dense > .5)
    adj_mat_sparse = adj_mat_dense.to_sparse()
    adj_mat_sparse = norm_adj_mat_two_node_types_sparse(adj_mat_sparse)
    adj_mat_dense = norm_adj_mat_two_node_types_dense(adj_mat_dense)
    assert torch.all(adj_mat_sparse.to_dense() - adj_mat_dense < 0.000001)


def test_norm_adj_mat_two_node_types_dense_01():
    adj_mat = torch.rand((10, 20))
    adj_mat = (adj_mat > .5)
    _ = norm_adj_mat_two_node_types_dense(adj_mat)


def test_norm_adj_mat_two_node_types_dense_02():
    adj_mat = torch.tensor([
        [0, 1, 1, 0], # 2
        [1, 0, 1, 0], # 2
        [1, 1, 0, 1], # 3
        [0, 0, 1, 0] # 1
    #    2  2  3  1
    ])
    expect_denom = np.array([
        [ 2,           2,           sqrt(6),   sqrt(2)   ],
        [ 2,           2,           sqrt(6),   sqrt(2)   ],
        [ sqrt(6),     sqrt(6),     3,         sqrt(3)   ],
        [ sqrt(2),     sqrt(2),     sqrt(3),   1         ]
    ], dtype=np.float32)
    expect = adj_mat.detach().cpu().numpy().astype(np.float32) / expect_denom
    res = decagon_pytorch.normalize.norm_adj_mat_two_node_types(adj_mat)
    res = res.todense().astype(np.float32)
    print('res:', res)
    print('expect:', expect)
    assert np.all(res - expect < 0.000001)


def test_norm_adj_mat_two_node_types_dense_03():
    adj_mat = torch.tensor([
        [0, 1, 1,  0,  0],
        [1, 0, 1,  0,  1],
        [1, 1, 0, .5, .5],
        [0, 0, .5, 0,  1],
        [0, 1, .5, 1,  0]
    ])
    adj_mat_dec = decagon_pytorch.normalize.norm_adj_mat_two_node_types(adj_mat)
    adj_mat_ico = norm_adj_mat_two_node_types_dense(adj_mat)
    adj_mat_dec = adj_mat_dec.todense()
    adj_mat_ico = adj_mat_ico.detach().cpu().numpy()
    print('adj_mat_dec:', adj_mat_dec)
    print('adj_mat_ico:', adj_mat_ico)
    assert np.all(adj_mat_dec - adj_mat_ico < 0.000001)


def test_norm_adj_mat_two_node_types_dense_04():
    adj_mat = torch.rand((10, 20))
    adj_mat_dec = decagon_pytorch.normalize.norm_adj_mat_two_node_types(adj_mat)
    adj_mat_ico = norm_adj_mat_two_node_types_dense(adj_mat)
    adj_mat_dec = adj_mat_dec.todense()
    adj_mat_ico = adj_mat_ico.detach().cpu().numpy()
    print('adj_mat_dec:', adj_mat_dec)
    print('adj_mat_ico:', adj_mat_ico)
    assert np.all(adj_mat_dec - adj_mat_ico < 0.000001)
