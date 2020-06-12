#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from .layer import Layer
import torch
from typing import Union, \
    List
from ..data import Data


class InputLayer(Layer):
    def __init__(self, data: Data, output_dim: Union[int, List[int]]= None, **kwargs) -> None:
        output_dim = output_dim or \
            list(map(lambda a: a.count, data.node_types))
        if not isinstance(output_dim, list):
            output_dim = [output_dim,] * len(data.node_types)

        super().__init__(output_dim, is_sparse=False, **kwargs)
        self.data = data
        self.node_reps = None
        self.build()

    def build(self) -> None:
        self.node_reps = []
        for i, nt in enumerate(self.data.node_types):
            reps = torch.rand(nt.count, self.output_dim[i])
            reps = torch.nn.Parameter(reps)
            self.register_parameter('node_reps[%d]' % i, reps)
            self.node_reps.append(reps)

    def forward(self) -> List[torch.nn.Parameter]:
        return self.node_reps

    def __repr__(self) -> str:
        s = ''
        s += 'GNN input layer with output_dim: %s\n' % self.output_dim
        s += '  # of node types: %d\n' % len(self.data.node_types)
        for nt in self.data.node_types:
            s += '    - %s (%d)\n' % (nt.name, nt.count)
        return s.strip()


class OneHotInputLayer(Layer):
    def __init__(self, data: Data, **kwargs) -> None:
        output_dim = [ a.count for a in data.node_types ]
        super().__init__(output_dim, is_sparse=True, **kwargs)
        self.data = data
        self.node_reps = None
        self.build()

    def build(self) -> None:
        self.node_reps = []
        for i, nt in enumerate(self.data.node_types):
            reps = torch.eye(nt.count).to_sparse()
            reps = torch.nn.Parameter(reps)
            self.register_parameter('node_reps[%d]' % i, reps)
            self.node_reps.append(reps)

    def forward(self) -> List[torch.nn.Parameter]:
        return self.node_reps

    def __repr__(self) -> str:
        s = ''
        s += 'One-hot GNN input layer\n'
        s += '  # of node types: %d\n' % len(self.data.node_types)
        for nt in self.data.node_types:
            s += '    - %s (%d)\n' % (nt.name, nt.count)
        return s.strip()
