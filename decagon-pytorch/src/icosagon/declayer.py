#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from .data import Data
from .trainprep import PreparedData, \
    TrainValTest
from typing import Type, \
    List, \
    Callable, \
    Union, \
    Dict, \
    Tuple
from .decode import DEDICOMDecoder


class DecodeLayer(torch.nn.Module):
    def __init__(self,
        input_dim: List[int],
        data: PreparedData,
        keep_prob: float = 1.,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        **kwargs) -> None:

        super().__init__(**kwargs)

        if not isinstance(input_dim, list):
            raise TypeError('input_dim must be a List')

        if not all([ a == input_dim[0] for a in input_dim ]):
            raise ValueError('All elements of input_dim must have the same value')

        if not isinstance(data, PreparedData):
            raise TypeError('data must be an instance of PreparedData')

        self.input_dim = input_dim
        self.output_dim = 1
        self.data = data
        self.keep_prob = keep_prob
        self.activation = activation

        self.decoders = None
        self.build()

    def build(self) -> None:
        self.decoders = []

        for fam in self.data.relation_families:
            for (node_type_row, node_type_column), rels in fam.relation_types.items():
                for r in rels:
                    pass

            dec = fam.decoder_class()
            self.decoders.append(dec)

        for (node_type_row, node_type_column), rels in self.data.relation_types.items():
            if len(rels) == 0:
                continue

            if isinstance(self.decoder_class, dict):
                if (node_type_row, node_type_column) in self.decoder_class:
                    decoder_class = self.decoder_class[node_type_row, node_type_column]
                elif (node_type_column, node_type_row) in self.decoder_class:
                    decoder_class = self.decoder_class[node_type_column, node_type_row]
                else:
                    raise KeyError('Decoder not specified for edge type: %s -- %s' % (
                        self.data.node_types[node_type_row].name,
                        self.data.node_types[node_type_column].name))
            else:
                decoder_class = self.decoder_class

            self.decoders[node_type_row, node_type_column] = \
                decoder_class(self.input_dim[node_type_row],
                    num_relation_types = len(rels),
                    keep_prob = self.keep_prob,
                    activation = self.activation)

    def forward(self, last_layer_repr: List[torch.Tensor]) -> Dict[Tuple[int, int], List[torch.Tensor]]:
        res = {}
        for (node_type_row, node_type_column), dec in self.decoders.items():
            inputs_row = last_layer_repr[node_type_row]
            inputs_column = last_layer_repr[node_type_column]
            pred_adj_matrices = [ dec(inputs_row, inputs_column, k) for k in range(dec.num_relation_types) ]
            res[node_type_row, node_type_column] = pred_adj_matrices
        return res
