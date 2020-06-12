from icosagon.dropout import dropout_sparse, \
    dropout_dense
import torch
import numpy as np


def test_dropout_01():
    for i in range(11):
        torch.random.manual_seed(i)
        a = torch.rand((5, 10))
        a[a < .5] = 0

        keep_prob=i/10. + np.finfo(np.float32).eps

        torch.random.manual_seed(i)
        b = dropout_dense(a, keep_prob=keep_prob)

        torch.random.manual_seed(i)
        c = dropout_sparse(a.to_sparse(), keep_prob=keep_prob)

        print('keep_prob:', keep_prob)
        print('a:', a.detach().cpu().numpy())
        print('b:', b.detach().cpu().numpy())
        print('c:', c, c.to_dense().detach().cpu().numpy())

        assert torch.all(b == c.to_dense())
