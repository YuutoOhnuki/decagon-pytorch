#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from .layer import Layer
import torch
from ..convolve import DropoutGraphConvActivation
from ..data import Data
from typing import List, \
    Union, \
    Callable
from collections import defaultdict


class DecagonLayer(Layer):
    def __init__(self,
        data: Data,
        previous_layer: Layer,
        output_dim: Union[int, List[int]],
        keep_prob: float = 1.,
        rel_activation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        layer_activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu,
        **kwargs):
        if not isinstance(output_dim, list):
            output_dim = [ output_dim ] * len(data.node_types)
        super().__init__(output_dim, is_sparse=False, **kwargs)
        self.data = data
        self.previous_layer = previous_layer
        self.input_dim = previous_layer.output_dim
        self.keep_prob = keep_prob
        self.rel_activation = rel_activation
        self.layer_activation = layer_activation
        self.next_layer_repr = None
        self.build()

    def build(self):
        self.next_layer_repr = defaultdict(list)

        for (nt_row, nt_col), relation_types in self.data.relation_types.items():
            row_convs = []
            col_convs = []

            for rel in relation_types:
                conv = DropoutGraphConvActivation(self.input_dim[nt_col],
                    self.output_dim[nt_row], rel.adjacency_matrix,
                    self.keep_prob, self.rel_activation)
                row_convs.append(conv)

                if nt_row == nt_col:
                    continue

                conv = DropoutGraphConvActivation(self.input_dim[nt_row],
                    self.output_dim[nt_col], rel.adjacency_matrix.transpose(0, 1),
                    self.keep_prob, self.rel_activation)
                col_convs.append(conv)

            self.next_layer_repr[nt_row].append((row_convs, nt_col))

            if nt_row == nt_col:
                continue

            self.next_layer_repr[nt_col].append((col_convs, nt_row))

    def __call__(self):
        prev_layer_repr = self.previous_layer()
        next_layer_repr = [ [] for _ in range(len(self.data.node_types)) ]
        print('next_layer_repr:', next_layer_repr)
        for i in range(len(self.data.node_types)):
            for convs, neighbor_type in self.next_layer_repr[i]:
                convs = [ conv(prev_layer_repr[neighbor_type]) \
                    for conv in convs ]
                convs = sum(convs)
                convs = torch.nn.functional.normalize(convs, p=2, dim=1)
                next_layer_repr[i].append(convs)
            next_layer_repr[i] = sum(next_layer_repr[i])
            next_layer_repr[i] = self.layer_activation(next_layer_repr[i])
        print('next_layer_repr:', next_layer_repr)
        return next_layer_repr
