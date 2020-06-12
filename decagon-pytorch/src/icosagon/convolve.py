#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from .dropout import dropout
from .weights import init_glorot
from typing import List, Callable


class GraphConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
        adjacency_matrix: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = init_glorot(in_channels, out_channels)
        self.adjacency_matrix = adjacency_matrix


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(x, self.weight) \
            if x.is_sparse \
            else torch.mm(x, self.weight)
        x = torch.sparse.mm(self.adjacency_matrix, x) \
            if self.adjacency_matrix.is_sparse \
            else torch.mm(self.adjacency_matrix, x)
        return x


class DropoutGraphConvActivation(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
        adjacency_matrix: torch.Tensor, keep_prob: float=1.,
        activation: Callable[[torch.Tensor], torch.Tensor]=torch.nn.functional.relu,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adjacency_matrix = adjacency_matrix
        self.keep_prob = keep_prob
        self.activation = activation
        self.graph_conv = GraphConv(input_dim, output_dim, adjacency_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = dropout(x, self.keep_prob)
        x = self.graph_conv(x)
        x = self.activation(x)
        return x
