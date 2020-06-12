#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from ..dropout import dropout
from ..weights import init_glorot
from typing import List, Callable


class DenseGraphConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
        adjacency_matrix: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = init_glorot(in_channels, out_channels)
        self.adjacency_matrix = adjacency_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adjacency_matrix, x)
        return x


class DenseDropoutGraphConvActivation(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
        adjacency_matrix: torch.Tensor, keep_prob: float=1.,
        activation: Callable[[torch.Tensor], torch.Tensor]=torch.nn.functional.relu,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.graph_conv = DenseGraphConv(input_dim, output_dim, adjacency_matrix)
        self.keep_prob = keep_prob
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = dropout(x, keep_prob=self.keep_prob)
        x = self.graph_conv(x)
        x = self.activation(x)
        return x


class DenseMultiDGCA(torch.nn.Module):
    def __init__(self, input_dim: List[int], output_dim: int,
        adjacency_matrices: List[torch.Tensor], keep_prob: float=1.,
        activation: Callable[[torch.Tensor], torch.Tensor]=torch.nn.functional.relu,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adjacency_matrices = adjacency_matrices
        self.keep_prob = keep_prob
        self.activation = activation
        self.dgca = None
        self.build()

    def build(self):
        if len(self.input_dim) != len(self.adjacency_matrices):
            raise ValueError('input_dim must have the same length as adjacency_matrices')
        self.dgca = []
        for input_dim, adj_mat in zip(self.input_dim, self.adjacency_matrices):
            self.dgca.append(DenseDropoutGraphConvActivation(input_dim, self.output_dim, adj_mat, self.keep_prob, self.activation))

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(x, list):
            raise ValueError('x must be a list of tensors')
        out = torch.zeros(len(x[0]), self.output_dim, dtype=x[0].dtype)
        for i, f in enumerate(self.dgca):
            out += f(x[i])
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out
