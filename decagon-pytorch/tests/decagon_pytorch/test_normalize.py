import decagon_pytorch.normalize
import decagon.deep.minibatch
import numpy as np


def test_normalize_adjacency_matrix_square():
    mx = np.random.rand(10, 10)
    mx[mx < .5] = 0
    mx = np.ceil(mx)
    res_torch = decagon_pytorch.normalize.normalize_adjacency_matrix(mx)
    res_tf = decagon.deep.minibatch.EdgeMinibatchIterator.preprocess_graph(None, mx)
    assert len(res_torch) == len(res_tf)
    for i in range(len(res_torch)):
        assert np.all(res_torch[i] == res_tf[i])


def test_normalize_adjacency_matrix_nonsquare():
    mx = np.random.rand(5, 10)
    mx[mx < .5] = 0
    mx = np.ceil(mx)
    res_torch = decagon_pytorch.normalize.normalize_adjacency_matrix(mx)
    res_tf = decagon.deep.minibatch.EdgeMinibatchIterator.preprocess_graph(None, mx)
    assert len(res_torch) == len(res_tf)
    for i in range(len(res_torch)):
        assert np.all(res_torch[i] == res_tf[i])
