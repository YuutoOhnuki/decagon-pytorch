from decagon_pytorch.layer import InputLayer, \
    OneHotInputLayer, \
    DecagonLayer
from decagon_pytorch.data import Data
import torch
import pytest
from decagon_pytorch.convolve import SparseDropoutGraphConvActivation, \
    SparseMultiDGCA, \
    DropoutGraphConvActivation


def _some_data():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    d.add_relation_type('Target', 1, 0, None)
    d.add_relation_type('Interaction', 0, 0, None)
    d.add_relation_type('Side Effect: Nausea', 1, 1, None)
    d.add_relation_type('Side Effect: Infertility', 1, 1, None)
    d.add_relation_type('Side Effect: Death', 1, 1, None)
    return d


def _some_data_with_interactions():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    d.add_relation_type('Target', 1, 0,
        torch.rand((100, 1000), dtype=torch.float32).round())
    d.add_relation_type('Interaction', 0, 0,
        torch.rand((1000, 1000), dtype=torch.float32).round())
    d.add_relation_type('Side Effect: Nausea', 1, 1,
        torch.rand((100, 100), dtype=torch.float32).round())
    d.add_relation_type('Side Effect: Infertility', 1, 1,
        torch.rand((100, 100), dtype=torch.float32).round())
    d.add_relation_type('Side Effect: Death', 1, 1,
        torch.rand((100, 100), dtype=torch.float32).round())
    return d


def test_decagon_layer_01():
    d = _some_data_with_interactions()
    in_layer = InputLayer(d)
    d_layer = DecagonLayer(d, in_layer, output_dim=32)


def test_decagon_layer_02():
    d = _some_data_with_interactions()
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(d, in_layer, output_dim=32)
    _ = d_layer() # dummy call


def test_decagon_layer_03():
    d = _some_data_with_interactions()
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(d, in_layer, output_dim=32)
    assert d_layer.data == d
    assert d_layer.previous_layer == in_layer
    assert d_layer.input_dim == [ 1000, 100 ]
    assert not d_layer.is_sparse
    assert d_layer.keep_prob == 1.
    assert d_layer.rel_activation(0.5) == 0.5
    x = torch.tensor([-1, 0, 0.5, 1])
    assert (d_layer.layer_activation(x) == torch.nn.functional.relu(x)).all()
    assert len(d_layer.next_layer_repr) == 2

    for i in range(2):
        assert len(d_layer.next_layer_repr[i]) == 2
        assert isinstance(d_layer.next_layer_repr[i], list)
        assert isinstance(d_layer.next_layer_repr[i][0], tuple)
        assert isinstance(d_layer.next_layer_repr[i][0][0], list)
        assert isinstance(d_layer.next_layer_repr[i][0][1], int)
        assert all([
            isinstance(dgca, DropoutGraphConvActivation) \
                for dgca in d_layer.next_layer_repr[i][0][0]
        ])
        assert all([
            dgca.output_dim == 32 \
                for dgca in d_layer.next_layer_repr[i][0][0]
        ])


def test_decagon_layer_04():
    # check if it is equivalent to MultiDGCA, as it should be

    d = Data()
    d.add_node_type('Dummy', 100)
    d.add_relation_type('Dummy Relation', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())

    in_layer = OneHotInputLayer(d)

    multi_dgca = SparseMultiDGCA([10], 32,
        [r.adjacency_matrix for r in d.relation_types[0, 0]],
        keep_prob=1., activation=lambda x: x)

    d_layer = DecagonLayer(d, in_layer, output_dim=32,
        keep_prob=1., rel_activation=lambda x: x,
        layer_activation=lambda x: x)

    assert isinstance(d_layer.next_layer_repr[0][0][0][0],
        DropoutGraphConvActivation)

    weight = d_layer.next_layer_repr[0][0][0][0].graph_conv.weight
    assert isinstance(weight, torch.Tensor)

    assert len(multi_dgca.sparse_dgca) == 1
    assert isinstance(multi_dgca.sparse_dgca[0], SparseDropoutGraphConvActivation)

    multi_dgca.sparse_dgca[0].sparse_graph_conv.weight = weight

    out_d_layer = d_layer()
    out_multi_dgca = multi_dgca(in_layer())

    assert isinstance(out_d_layer, list)
    assert len(out_d_layer) == 1

    assert torch.all(out_d_layer[0] == out_multi_dgca)


def test_decagon_layer_05():
    # check if it is equivalent to MultiDGCA, as it should be
    # this time for two relations, same edge type

    d = Data()
    d.add_node_type('Dummy', 100)
    d.add_relation_type('Dummy Relation 1', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())
    d.add_relation_type('Dummy Relation 2', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())

    in_layer = OneHotInputLayer(d)

    multi_dgca = SparseMultiDGCA([100, 100], 32,
        [r.adjacency_matrix for r in d.relation_types[0, 0]],
        keep_prob=1., activation=lambda x: x)

    d_layer = DecagonLayer(d, in_layer, output_dim=32,
        keep_prob=1., rel_activation=lambda x: x,
        layer_activation=lambda x: x)

    assert all([
        isinstance(dgca, DropoutGraphConvActivation) \
            for dgca in d_layer.next_layer_repr[0][0][0]
    ])

    weight = [ dgca.graph_conv.weight \
        for dgca in d_layer.next_layer_repr[0][0][0] ]
    assert all([
        isinstance(w, torch.Tensor) \
            for w in weight
    ])

    assert len(multi_dgca.sparse_dgca) == 2
    for i in range(2):
        assert isinstance(multi_dgca.sparse_dgca[i], SparseDropoutGraphConvActivation)

    for i in range(2):
        multi_dgca.sparse_dgca[i].sparse_graph_conv.weight = weight[i]

    out_d_layer = d_layer()
    x = in_layer()
    out_multi_dgca = multi_dgca([ x[0], x[0] ])

    assert isinstance(out_d_layer, list)
    assert len(out_d_layer) == 1

    assert torch.all(out_d_layer[0] == out_multi_dgca)
