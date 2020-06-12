from icosagon.convolve import GraphConv, \
    DropoutGraphConvActivation
import torch
from icosagon.dropout import dropout


def _test_graph_conv_01(use_sparse: bool):
    adj_mat = torch.rand((10, 20))
    adj_mat[adj_mat < .5] = 0
    adj_mat = torch.ceil(adj_mat)

    node_reprs = torch.eye(20)

    graph_conv = GraphConv(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    graph_conv.weight = torch.eye(20)

    res = graph_conv(node_reprs)
    assert torch.all(res == adj_mat)


def _test_graph_conv_02(use_sparse: bool):
    adj_mat = torch.rand((10, 20))
    adj_mat[adj_mat < .5] = 0
    adj_mat = torch.ceil(adj_mat)

    node_reprs = torch.eye(20)

    graph_conv = GraphConv(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    graph_conv.weight = torch.eye(20) * 2

    res = graph_conv(node_reprs)
    assert torch.all(res == adj_mat * 2)


def _test_graph_conv_03(use_sparse: bool):
    adj_mat = torch.tensor([
        [1, 0, 1, 0, 1, 0], # [1, 0, 0]
        [1, 0, 1, 0, 0, 1], # [1, 0, 0]
        [1, 1, 0, 1, 0, 0], # [0, 1, 0]
        [0, 0, 0, 1, 0, 1], # [0, 1, 0]
        [1, 1, 1, 1, 1, 1], # [0, 0, 1]
        [0, 0, 0, 1, 1, 1]  # [0, 0, 1]
    ], dtype=torch.float32)

    expect = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
        [2, 1, 0],
        [0, 1, 1],
        [2, 2, 2],
        [0, 1, 2]
    ], dtype=torch.float32)

    node_reprs = torch.eye(6)

    graph_conv = GraphConv(6, 3, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    graph_conv.weight = torch.tensor([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=torch.float32)

    res = graph_conv(node_reprs)
    assert torch.all(res == expect)


def test_graph_conv_dense_01():
    _test_graph_conv_01(use_sparse=False)


def test_graph_conv_dense_02():
    _test_graph_conv_02(use_sparse=False)


def test_graph_conv_dense_03():
    _test_graph_conv_03(use_sparse=False)


def test_graph_conv_sparse_01():
    _test_graph_conv_01(use_sparse=True)


def test_graph_conv_sparse_02():
    _test_graph_conv_02(use_sparse=True)


def test_graph_conv_sparse_03():
    _test_graph_conv_03(use_sparse=True)


def _test_dropout_graph_conv_activation_01(use_sparse: bool):
    adj_mat = torch.rand((10, 20))
    adj_mat[adj_mat < .5] = 0
    adj_mat = torch.ceil(adj_mat)
    node_reprs = torch.eye(20)

    conv_1 = DropoutGraphConvActivation(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat, keep_prob=1.,
        activation=lambda x: x)

    conv_2 = GraphConv(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    conv_2.weight = conv_1.graph_conv.weight

    res_1 = conv_1(node_reprs)
    res_2 = conv_2(node_reprs)

    print('res_1:', res_1.detach().cpu().numpy())
    print('res_2:', res_2.detach().cpu().numpy())

    assert torch.all(res_1 == res_2)


def _test_dropout_graph_conv_activation_02(use_sparse: bool):
    adj_mat = torch.rand((10, 20))
    adj_mat[adj_mat < .5] = 0
    adj_mat = torch.ceil(adj_mat)
    node_reprs = torch.eye(20)

    conv_1 = DropoutGraphConvActivation(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat, keep_prob=1.,
        activation=lambda x: x * 2)

    conv_2 = GraphConv(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    conv_2.weight = conv_1.graph_conv.weight

    res_1 = conv_1(node_reprs)
    res_2 = conv_2(node_reprs)

    print('res_1:', res_1.detach().cpu().numpy())
    print('res_2:', res_2.detach().cpu().numpy())

    assert torch.all(res_1 == res_2 * 2)


def _test_dropout_graph_conv_activation_03(use_sparse: bool):
    adj_mat = torch.rand((10, 20))
    adj_mat[adj_mat < .5] = 0
    adj_mat = torch.ceil(adj_mat)
    node_reprs = torch.eye(20)

    conv_1 = DropoutGraphConvActivation(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat, keep_prob=.5,
        activation=lambda x: x)

    conv_2 = GraphConv(20, 20, adj_mat.to_sparse() \
        if use_sparse else adj_mat)
    conv_2.weight = conv_1.graph_conv.weight

    torch.random.manual_seed(0)
    res_1 = conv_1(node_reprs)

    torch.random.manual_seed(0)
    res_2 = conv_2(dropout(node_reprs, 0.5))

    print('res_1:', res_1.detach().cpu().numpy())
    print('res_2:', res_2.detach().cpu().numpy())

    assert torch.all(res_1 == res_2)


def test_dropout_graph_conv_activation_dense_01():
    _test_dropout_graph_conv_activation_01(False)


def test_dropout_graph_conv_activation_sparse_01():
    _test_dropout_graph_conv_activation_01(True)


def test_dropout_graph_conv_activation_dense_02():
    _test_dropout_graph_conv_activation_02(False)


def test_dropout_graph_conv_activation_sparse_02():
    _test_dropout_graph_conv_activation_02(True)


def test_dropout_graph_conv_activation_dense_03():
    _test_dropout_graph_conv_activation_03(False)


def test_dropout_graph_conv_activation_sparse_03():
    _test_dropout_graph_conv_activation_03(True)
