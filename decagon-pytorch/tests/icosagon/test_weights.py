from icosagon.weights import init_glorot
import torch
import numpy as np


def test_init_glorot_01():
    torch.random.manual_seed(0)
    res = init_glorot(10, 20)
    torch.random.manual_seed(0)
    rnd = torch.rand((10, 20))
    init_range = np.sqrt(6.0 / 30)
    expected = -init_range + 2 * init_range * rnd
    assert torch.all(res == expected)


def test_init_glorot_02():
    torch.random.manual_seed(0)
    res = init_glorot(20, 10)
    torch.random.manual_seed(0)
    rnd = torch.rand((20, 10))
    init_range = np.sqrt(6.0 / 30)
    expected = -init_range + 2 * init_range * rnd
    assert torch.all(res == expected)
