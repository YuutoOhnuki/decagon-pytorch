#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch


def dropout_sparse(x, keep_prob):
    """Dropout for sparse tensors.
    """
    x = x.coalesce()
    i = x._indices()
    v = x._values()
    size = x.size()

    n = keep_prob + torch.rand(len(v))
    n = torch.floor(n).to(torch.bool)
    i = i[:,n]
    v = v[n]
    x = torch.sparse_coo_tensor(i, v, size=size)

    return x * (1./keep_prob)


def dropout(x, keep_prob):
    """Dropout for dense tensors.
    """
    shape = x.shape
    x = torch.flatten(x)
    n = keep_prob + torch.rand(len(x))
    n = (1. - torch.floor(n)).to(torch.bool)
    x[n] = 0
    x = torch.reshape(x, shape)
    # x = torch.nn.functional.dropout(x, p=1.-keep_prob)
    return x * (1./keep_prob)
