#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from .weights import init_glorot
from .dropout import dropout


class DEDICOMDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.global_interaction = init_glorot(input_dim, input_dim)
        self.local_variation = [
            torch.flatten(init_glorot(input_dim, 1)) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        relation = torch.diag(self.local_variation[relation_index])

        product1 = torch.mm(inputs_row, relation)
        product2 = torch.mm(product1, self.global_interaction)
        product3 = torch.mm(product2, relation)
        rec = torch.bmm(product3.view(product3.shape[0], 1, product3.shape[1]),
            inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class DistMultDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.relation = [
            torch.flatten(init_glorot(input_dim, 1)) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        relation = torch.diag(self.relation[relation_index])

        intermediate_product = torch.mm(inputs_row, relation)
        rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
            inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class BilinearDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.relation = [
            init_glorot(input_dim, input_dim) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        intermediate_product = torch.mm(inputs_row, self.relation[relation_index])
        rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
            inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class InnerProductDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation


    def forward(self, inputs_row, inputs_col, _):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        rec = torch.bmm(inputs_row.view(inputs_row.shape[0], 1, inputs_row.shape[1]),
            inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)
