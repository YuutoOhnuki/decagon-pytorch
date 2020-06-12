from decagon_pytorch.dropout import dropout_sparse
import torch
import numpy as np


def dropout_dense(a, keep_prob):
    i = np.array(np.where(a))
    v = a[i[0, :], i[1, :]]

    # torch.random.manual_seed(0)
    n = keep_prob + torch.rand(len(v))
    n = torch.floor(n).to(torch.bool)
    i = i[:, n]
    v = v[n]
    x = torch.sparse_coo_tensor(i, v, size=a.shape)

    return x * (1./keep_prob)


def test_dropout_sparse():
    for i in range(11):
        torch.random.manual_seed(i)
        a = torch.rand((5, 10))
        a[a < .5] = 0

        keep_prob=i/10. + np.finfo(np.float32).eps

        torch.random.manual_seed(i)
        b = dropout_dense(a, keep_prob=keep_prob)

        torch.random.manual_seed(i)
        c = dropout_sparse(a.to_sparse(), keep_prob=keep_prob)

        assert np.all(np.array(b.to_dense()) == np.array(c.to_dense()))
