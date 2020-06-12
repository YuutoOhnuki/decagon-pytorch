#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from typing import List, \
    Union


class Layer(torch.nn.Module):
    def __init__(self,
        output_dim: Union[int, List[int]],
        is_sparse: bool,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.is_sparse = is_sparse
