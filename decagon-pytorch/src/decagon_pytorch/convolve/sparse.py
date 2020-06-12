#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from ..dropout import dropout_sparse
from ..weights import init_glorot
from typing import List, Callable


class SparseGraphConv(torch.nn.Module):
    """Convolution layer for sparse inputs."""
    def __init__(self, in_channels: int, out_channels: int,
        adjacency_matrix: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = init_glorot(in_channels, out_channels)
        self.adjacency_matrix = adjacency_matrix


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(x, self.weight)
        x = torch.sparse.mm(self.adjacency_matrix, x)
        return x


class SparseDropoutGraphConvActivation(torch.nn.Module):
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
        self.sparse_graph_conv = SparseGraphConv(input_dim, output_dim, adjacency_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = dropout_sparse(x, self.keep_prob)
        x = self.sparse_graph_conv(x)
        x = self.activation(x)
        return x


class SparseMultiDGCA(torch.nn.Module):
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
        self.sparse_dgca = None
        self.build()

    def build(self):
        if len(self.input_dim) != len(self.adjacency_matrices):
            raise ValueError('input_dim must have the same length as adjacency_matrices')
        self.sparse_dgca = []
        for input_dim, adj_mat in zip(self.input_dim, self.adjacency_matrices):
            self.sparse_dgca.append(SparseDropoutGraphConvActivation(input_dim, self.output_dim, adj_mat, self.keep_prob, self.activation))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, list):
            raise ValueError('x must be a list of tensors')
        out = torch.zeros(len(x[0]), self.output_dim, dtype=x[0].dtype)
        for i, f in enumerate(self.sparse_dgca):
            out += f(x[i])
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out
