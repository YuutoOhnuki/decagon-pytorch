#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from .layer import Layer
import torch
from ..data import Data
from typing import Type, \
    List, \
    Callable, \
    Union, \
    Dict, \
    Tuple
from ..decode.cartesian import DEDICOMDecoder


class DecodeLayer(torch.nn.Module):
    def __init__(self,
        data: Data,
        last_layer: Layer,
        keep_prob: float = 1.,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        decoder_class: Union[Type, Dict[Tuple[int, int], Type]] = DEDICOMDecoder, **kwargs) -> None:

        super().__init__(**kwargs)
        self.data = data
        self.last_layer = last_layer
        self.keep_prob = keep_prob
        self.activation = activation
        assert all([a == last_layer.output_dim[0] \
            for a in last_layer.output_dim])
        self.input_dim = last_layer.output_dim[0]
        self.output_dim = 1
        self.decoder_class = decoder_class
        self.decoders = None
        self.build()

    def build(self) -> None:
        self.decoders = {}
        for (node_type_row, node_type_col), rels in self.data.relation_types.items():
            key = (node_type_row, node_type_col)
            if isinstance(self.decoder_class, dict):
                if key in self.decoder_class:
                    decoder_class = self.decoder_class[key]
                else:
                    raise KeyError('Decoder not specified for edge type: %d -- %d' % key)
            else:
                decoder_class = self.decoder_class

            self.decoders[key] = decoder_class(self.input_dim,
                num_relation_types = len(rels),
                drop_prob = 1. - self.keep_prob,
                activation = self.activation)


    def forward(self, last_layer_repr: List[torch.Tensor]):
        res = {}
        for (node_type_row, node_type_col), rel in self.data.relation_types.items():
            key = (node_type_row, node_type_col)
            inputs_row = last_layer_repr[node_type_row]
            inputs_col = last_layer_repr[node_type_col]
            pred_adj_matrices = self.decoders[key](inputs_row, inputs_col)
            res[node_type_row, node_type_col] = pred_adj_matrices
        return res
