import torch
from .convolve import DropoutGraphConvActivation
from .data import Data
from .trainprep import PreparedData
from typing import List, \
    Union, \
    Callable
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Convolutions(object):
    node_type_column: int
    convolutions: List[DropoutGraphConvActivation]


class DecagonLayer(torch.nn.Module):
    def __init__(self,
        input_dim: List[int],
        output_dim: List[int],
        data: Union[Data, PreparedData],
        keep_prob: float = 1.,
        rel_activation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        layer_activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu,
        **kwargs):

        super().__init__(**kwargs)

        if not isinstance(input_dim, list):
            raise ValueError('input_dim must be a list')

        if not output_dim:
            raise ValueError('output_dim must be specified')

        if not isinstance(output_dim, list):
            output_dim = [output_dim] * len(data.node_types)

        if not isinstance(data, Data) and not isinstance(data, PreparedData):
            raise ValueError('data must be of type Data or PreparedData')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.data = data
        self.keep_prob = float(keep_prob)
        self.rel_activation = rel_activation
        self.layer_activation = layer_activation

        self.is_sparse = False
        self.next_layer_repr = None
        self.build()

    def build_fam_one_node_type(self, fam):
        convolutions = []

        for r in fam.relation_types:
            conv = DropoutGraphConvActivation(self.input_dim[fam.node_type_column],
                self.output_dim[fam.node_type_row], r.adjacency_matrix,
                self.keep_prob, self.rel_activation)
            convolutions.append(conv)

        self.next_layer_repr[fam.node_type_row].append(
            Convolutions(fam.node_type_column, convolutions))

    def build_fam_two_node_types(self, fam) -> None:
        convolutions_row = []
        convolutions_column = []

        for r in fam.relation_types:
            if r.adjacency_matrix is not None:
                conv = DropoutGraphConvActivation(self.input_dim[fam.node_type_column],
                    self.output_dim[fam.node_type_row], r.adjacency_matrix,
                    self.keep_prob, self.rel_activation)
                convolutions_row.append(conv)

            if r.adjacency_matrix_backward is not None:
                conv = DropoutGraphConvActivation(self.input_dim[fam.node_type_row],
                    self.output_dim[fam.node_type_column], r.adjacency_matrix_backward,
                    self.keep_prob, self.rel_activation)
                convolutions_column.append(conv)

        self.next_layer_repr[fam.node_type_row].append(
            Convolutions(fam.node_type_column, convolutions_row))

        self.next_layer_repr[fam.node_type_column].append(
            Convolutions(fam.node_type_row, convolutions_column))

    def build_family(self, fam) -> None:
        if fam.node_type_row == fam.node_type_column:
            self.build_fam_one_node_type(fam)
        else:
            self.build_fam_two_node_types(fam)

    def build(self):
        self.next_layer_repr = [ [] for _ in range(len(self.data.node_types)) ]
        for fam in self.data.relation_families:
            self.build_family(fam)

    def __call__(self, prev_layer_repr):
        next_layer_repr = [ [] for _ in range(len(self.data.node_types)) ]
        n = len(self.data.node_types)

        for node_type_row in range(n):
            for convolutions in self.next_layer_repr[node_type_row]:
                repr_ = [ conv(prev_layer_repr[convolutions.node_type_column]) \
                    for conv in convolutions.convolutions ]
                repr_ = sum(repr_)
                repr_ = torch.nn.functional.normalize(repr_, p=2, dim=1)
                next_layer_repr[node_type_row].append(repr_)
            if len(next_layer_repr[node_type_row]) == 0:
                next_layer_repr[node_type_row] = torch.zeros(self.output_dim[node_type_row])
            else:
                next_layer_repr[node_type_row] = sum(next_layer_repr[node_type_row])
                next_layer_repr[node_type_row] = self.layer_activation(next_layer_repr[node_type_row])

        return next_layer_repr
