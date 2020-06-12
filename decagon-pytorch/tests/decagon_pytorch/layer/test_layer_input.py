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


def test_input_layer_01():
    d = _some_data()
    for output_dim in [32, 64, 128]:
        layer = InputLayer(d, output_dim)
        assert layer.output_dim[0] == output_dim
        assert len(layer.node_reps) == 2
        assert layer.node_reps[0].shape == (1000, output_dim)
        assert layer.node_reps[1].shape == (100, output_dim)
        assert layer.data == d


def test_input_layer_02():
    d = _some_data()
    layer = InputLayer(d, 32)
    res = layer()
    assert isinstance(res[0], torch.Tensor)
    assert isinstance(res[1], torch.Tensor)
    assert res[0].shape == (1000, 32)
    assert res[1].shape == (100, 32)
    assert torch.all(res[0] == layer.node_reps[0])
    assert torch.all(res[1] == layer.node_reps[1])


def test_input_layer_03():
    if torch.cuda.device_count() == 0:
        pytest.skip('No CUDA devices on this host')
    d = _some_data()
    layer = InputLayer(d, 32)
    device = torch.device('cuda:0')
    layer = layer.to(device)
    print(list(layer.parameters()))
    # assert layer.device.type == 'cuda:0'
    assert layer.node_reps[0].device == device
    assert layer.node_reps[1].device == device


def test_one_hot_input_layer_01():
    d = _some_data()
    layer = OneHotInputLayer(d)
    assert layer.output_dim == [1000, 100]
    assert len(layer.node_reps) == 2
    assert layer.node_reps[0].shape == (1000, 1000)
    assert layer.node_reps[1].shape == (100, 100)
    assert layer.data == d
    assert layer.is_sparse


def test_one_hot_input_layer_02():
    d = _some_data()
    layer = OneHotInputLayer(d)
    res = layer()
    assert isinstance(res[0], torch.Tensor)
    assert isinstance(res[1], torch.Tensor)
    assert res[0].shape == (1000, 1000)
    assert res[1].shape == (100, 100)
    assert torch.all(res[0].to_dense() == layer.node_reps[0].to_dense())
    assert torch.all(res[1].to_dense() == layer.node_reps[1].to_dense())


def test_one_hot_input_layer_03():
    if torch.cuda.device_count() == 0:
        pytest.skip('No CUDA devices on this host')
    d = _some_data()
    layer = OneHotInputLayer(d)
    device = torch.device('cuda:0')
    layer = layer.to(device)
    print(list(layer.parameters()))
    # assert layer.device.type == 'cuda:0'
    assert layer.node_reps[0].device == device
    assert layer.node_reps[1].device == device
