from icosagon.input import InputLayer, \
    OneHotInputLayer
from icosagon.convlayer import DecagonLayer, \
    Convolutions
from icosagon.data import Data
import torch
import pytest
from icosagon.convolve import DropoutGraphConvActivation
from decagon_pytorch.convolve import MultiDGCA
import decagon_pytorch.convolve


def _make_symmetric(x: torch.Tensor):
    x = (x + x.transpose(0, 1)) / 2
    return x


def _symmetric_random(n_rows, n_columns):
    return _make_symmetric(torch.rand((n_rows, n_columns),
        dtype=torch.float32).round())


def _some_data_with_interactions():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)

    fam = d.add_relation_family('Drug-Gene', 1, 0, True)
    fam.add_relation_type('Target', 1, 0,
        torch.rand((100, 1000), dtype=torch.float32).round())

    fam = d.add_relation_family('Gene-Gene', 0, 0, True)
    fam.add_relation_type('Interaction', 0, 0,
        _symmetric_random(1000, 1000))

    fam = d.add_relation_family('Drug-Drug', 1, 1, True)
    fam.add_relation_type('Side Effect: Nausea', 1, 1,
        _symmetric_random(100, 100))
    fam.add_relation_type('Side Effect: Infertility', 1, 1,
        _symmetric_random(100, 100))
    fam.add_relation_type('Side Effect: Death', 1, 1,
        _symmetric_random(100, 100))
    return d


def test_decagon_layer_01():
    d = _some_data_with_interactions()
    in_layer = InputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    seq = torch.nn.Sequential(in_layer, d_layer)
    _ = seq(None) # dummy call


def test_decagon_layer_02():
    d = _some_data_with_interactions()
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    seq = torch.nn.Sequential(in_layer, d_layer)
    _ = seq(None) # dummy call


def test_decagon_layer_03():
    d = _some_data_with_interactions()
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)

    assert d_layer.input_dim == [ 1000, 100 ]
    assert d_layer.output_dim == [ 32, 32 ]
    assert d_layer.data == d
    assert d_layer.keep_prob == 1.
    assert d_layer.rel_activation(0.5) == 0.5
    x = torch.tensor([-1, 0, 0.5, 1])
    assert (d_layer.layer_activation(x) == torch.nn.functional.relu(x)).all()

    assert not d_layer.is_sparse
    assert len(d_layer.next_layer_repr) == 2

    for i in range(2):
        assert len(d_layer.next_layer_repr[i]) == 2
        assert isinstance(d_layer.next_layer_repr[i], list)
        assert isinstance(d_layer.next_layer_repr[i][0], Convolutions)
        assert isinstance(d_layer.next_layer_repr[i][0].node_type_column, int)
        assert isinstance(d_layer.next_layer_repr[i][0].convolutions, list)
        assert all([
            isinstance(dgca, DropoutGraphConvActivation) \
                for dgca in d_layer.next_layer_repr[i][0].convolutions
        ])
        assert all([
            dgca.output_dim == 32 \
                for dgca in d_layer.next_layer_repr[i][0].convolutions
        ])


def test_decagon_layer_04():
    # check if it is equivalent to MultiDGCA, as it should be

    d = Data()
    d.add_node_type('Dummy', 100)
    fam = d.add_relation_family('Dummy-Dummy', 0, 0, True)
    fam.add_relation_type('Dummy Relation', 0, 0,
        _symmetric_random(100, 100).to_sparse())

    in_layer = OneHotInputLayer(d)

    multi_dgca = MultiDGCA([10], 32,
        [r.adjacency_matrix for r in fam.relation_types],
        keep_prob=1., activation=lambda x: x)

    d_layer = DecagonLayer(in_layer.output_dim, 32, d,
        keep_prob=1., rel_activation=lambda x: x,
        layer_activation=lambda x: x)

    assert isinstance(d_layer.next_layer_repr[0][0].convolutions[0],
        DropoutGraphConvActivation)

    weight = d_layer.next_layer_repr[0][0].convolutions[0].graph_conv.weight
    assert isinstance(weight, torch.Tensor)

    assert len(multi_dgca.dgca) == 1
    assert isinstance(multi_dgca.dgca[0],
        decagon_pytorch.convolve.DropoutGraphConvActivation)

    multi_dgca.dgca[0].graph_conv.weight = weight

    seq = torch.nn.Sequential(in_layer, d_layer)
    out_d_layer = seq(None)
    out_multi_dgca = multi_dgca(in_layer(None))

    assert isinstance(out_d_layer, list)
    assert len(out_d_layer) == 1

    assert torch.all(out_d_layer[0] == out_multi_dgca)


def test_decagon_layer_05():
    # check if it is equivalent to MultiDGCA, as it should be
    # this time for two relations, same edge type

    d = Data()
    d.add_node_type('Dummy', 100)
    fam = d.add_relation_family('Dummy-Dummy', 0, 0, True)
    fam.add_relation_type('Dummy Relation 1', 0, 0,
        _symmetric_random(100, 100).to_sparse())
    fam.add_relation_type('Dummy Relation 2', 0, 0,
        _symmetric_random(100, 100).to_sparse())

    in_layer = OneHotInputLayer(d)

    multi_dgca = MultiDGCA([100, 100], 32,
        [r.adjacency_matrix for r in fam.relation_types],
        keep_prob=1., activation=lambda x: x)

    d_layer = DecagonLayer(in_layer.output_dim, output_dim=32, data=d,
        keep_prob=1., rel_activation=lambda x: x,
        layer_activation=lambda x: x)

    assert all([
        isinstance(dgca, DropoutGraphConvActivation) \
            for dgca in d_layer.next_layer_repr[0][0].convolutions
    ])

    weight = [ dgca.graph_conv.weight \
        for dgca in d_layer.next_layer_repr[0][0].convolutions ]
    assert all([
        isinstance(w, torch.Tensor) \
            for w in weight
    ])

    assert len(multi_dgca.dgca) == 2
    for i in range(2):
        assert isinstance(multi_dgca.dgca[i],
            decagon_pytorch.convolve.DropoutGraphConvActivation)

    for i in range(2):
        multi_dgca.dgca[i].graph_conv.weight = weight[i]

    seq = torch.nn.Sequential(in_layer, d_layer)
    out_d_layer = seq(None)
    x = in_layer(None)
    out_multi_dgca = multi_dgca([ x[0], x[0] ])

    assert isinstance(out_d_layer, list)
    assert len(out_d_layer) == 1

    assert torch.all(out_d_layer[0] == out_multi_dgca)
