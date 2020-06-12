#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from icosagon.input import OneHotInputLayer
from icosagon.convlayer import DecagonLayer
from icosagon.declayer import DecodeLayer
from icosagon.decode import DEDICOMDecoder
from icosagon.data import Data
from icosagon.trainprep import prepare_training, \
    TrainValTest
import torch


def test_decode_layer_01():
    d = Data()
    d.add_node_type('Dummy', 100)
    fam = d.add_relation_family('Dummy-Dummy', 0, 0, False)
    fam.add_relation_type('Dummy Relation 1', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())
    prep_d = prepare_training(d, TrainValTest(.8, .1, .1))
    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    seq = torch.nn.Sequential(in_layer, d_layer)
    last_layer_repr = seq(None)
    dec = DecodeLayer(input_dim=d_layer.output_dim, data=prep_d, keep_prob=1.,
        decoder_class=DEDICOMDecoder, activation=lambda x: x)
    pred_adj_matrices = dec(last_layer_repr)
    assert isinstance(pred_adj_matrices, dict)
    assert len(pred_adj_matrices) == 1
    assert isinstance(pred_adj_matrices[0, 0], list)
    assert len(pred_adj_matrices[0, 0]) == 1


def test_decode_layer_02():
    d = Data()
    d.add_node_type('Dummy', 100)
    d.add_relation_type('Dummy Relation 1', 0, 0,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())
    prep_d = prepare_training(d, TrainValTest(.8, .1, .1))

    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    dec_layer = DecodeLayer(input_dim=d_layer.output_dim, data=prep_d, keep_prob=1.,
        decoder_class=DEDICOMDecoder, activation=lambda x: x)
    seq = torch.nn.Sequential(in_layer, d_layer, dec_layer)

    pred_adj_matrices = seq(None)

    assert isinstance(pred_adj_matrices, dict)
    assert len(pred_adj_matrices) == 1
    assert isinstance(pred_adj_matrices[0, 0], list)
    assert len(pred_adj_matrices[0, 0]) == 1


def test_decode_layer_03():
    d = Data()
    d.add_node_type('Dummy 1', 100)
    d.add_node_type('Dummy 2', 100)
    d.add_relation_type('Dummy Relation 1', 0, 1,
        torch.rand((100, 100), dtype=torch.float32).round().to_sparse())
    prep_d = prepare_training(d, TrainValTest(.8, .1, .1))

    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    dec_layer = DecodeLayer(input_dim=d_layer.output_dim, data=prep_d, keep_prob=1.,
        decoder_class={(0, 1): DEDICOMDecoder}, activation=lambda x: x)
    seq = torch.nn.Sequential(in_layer, d_layer, dec_layer)

    pred_adj_matrices = seq(None)
    assert isinstance(pred_adj_matrices, dict)
    assert len(pred_adj_matrices) == 2
    assert isinstance(pred_adj_matrices[0, 1], list)
    assert isinstance(pred_adj_matrices[1, 0], list)
    assert len(pred_adj_matrices[0, 1]) == 1
    assert len(pred_adj_matrices[1, 0]) == 1


def test_decode_layer_04():
    d = Data()
    d.add_node_type('Dummy', 100)
    assert len(d.relation_types[0, 0]) == 0

    prep_d = prepare_training(d, TrainValTest(.8, .1, .1))

    in_layer = OneHotInputLayer(d)
    d_layer = DecagonLayer(in_layer.output_dim, 32, d)
    dec_layer = DecodeLayer(input_dim=d_layer.output_dim, data=prep_d, keep_prob=1.,
        decoder_class=DEDICOMDecoder, activation=lambda x: x)
    seq = torch.nn.Sequential(in_layer, d_layer, dec_layer)

    pred_adj_matrices = seq(None)

    assert isinstance(pred_adj_matrices, dict)
    assert len(pred_adj_matrices) == 0
