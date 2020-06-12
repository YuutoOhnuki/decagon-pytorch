from decagon_pytorch.layer import OneHotInputLayer, \
    DecagonLayer, \
    DecodeLayer
from decagon_pytorch.decode.cartesian import DEDICOMDecoder
from decagon_pytorch.data import Data
import torch


def test_decode_layer_01():
    d = Data()
    d.add_node_type('Dummy', 100)
    d.add_relation_type('Dummy Relation 1', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(d, in_layer, 32)
    last_layer_repr = d_layer()
    dec = DecodeLayer(d, last_layer = d_layer, decoder_class = DEDICOMDecoder)
    pred_adj_matrices = dec(last_layer_repr)
    assert isinstance(pred_adj_matrices, dict)
    assert len(pred_adj_matrices) == 1
    assert isinstance(pred_adj_matrices[0, 0], list)
    assert len(pred_adj_matrices[0, 0]) == 1
