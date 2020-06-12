import decagon_pytorch.convolve
import decagon.deep.layers
import torch
import tensorflow as tf
import numpy as np


def prepare_data():
    np.random.seed(0)
    latent = np.random.random((5, 10)).astype(np.float32)
    latent[latent < .5] = 0
    latent = np.ceil(latent)
    adjacency_matrices = []
    for _ in range(5):
        adj_mat = np.random.random((len(latent),) * 2).astype(np.float32)
        adj_mat[adj_mat < .5] = 0
        adj_mat = np.ceil(adj_mat)
        adjacency_matrices.append(adj_mat)
    print('latent:', latent)
    print('adjacency_matrices[0]:', adjacency_matrices[0])
    return latent, adjacency_matrices


def dense_to_sparse_tf(x):
    a, b = np.where(x)
    indices = np.array([a, b]).T
    values = x[a, b]
    return tf.sparse.SparseTensor(indices, values, x.shape)


def dropout_sparse_tf(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.convert_to_tensor(torch.rand(noise_shape).detach().numpy())
    # tf.convert_to_tensor(np.random.random(noise_shape))
    # tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dense_graph_conv_torch():
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    latent = torch.tensor(latent)
    adj_mat = adjacency_matrices[0]
    adj_mat = torch.tensor(adj_mat)
    conv = decagon_pytorch.convolve.DenseGraphConv(10, 10,
        adj_mat)
    latent = conv(latent)
    return latent


def dense_dropout_graph_conv_activation_torch(keep_prob=1.):
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    latent = torch.tensor(latent)
    adj_mat = adjacency_matrices[0]
    adj_mat = torch.tensor(adj_mat)
    conv = decagon_pytorch.convolve.DenseDropoutGraphConvActivation(10, 10,
        adj_mat, keep_prob=keep_prob)
    latent = conv(latent)
    return latent


def sparse_graph_conv_torch():
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    print('latent.dtype:', latent.dtype)
    latent = torch.tensor(latent).to_sparse()
    adj_mat = adjacency_matrices[0]
    adj_mat = torch.tensor(adj_mat).to_sparse()
    print('adj_mat.dtype:', adj_mat.dtype,
        'latent.dtype:', latent.dtype)
    conv = decagon_pytorch.convolve.SparseGraphConv(10, 10,
        adj_mat)
    latent = conv(latent)
    return latent


def sparse_graph_conv_tf():
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    conv_torch = decagon_pytorch.convolve.SparseGraphConv(10, 10,
        torch.tensor(adjacency_matrices[0]).to_sparse())
    weight = tf.constant(conv_torch.weight.detach().numpy())
    latent = dense_to_sparse_tf(latent)
    adj_mat = dense_to_sparse_tf(adjacency_matrices[0])
    latent = tf.sparse_tensor_dense_matmul(latent, weight)
    latent = tf.sparse_tensor_dense_matmul(adj_mat, latent)
    return latent


def sparse_dropout_graph_conv_activation_torch(keep_prob=1.):
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    latent = torch.tensor(latent).to_sparse()
    adj_mat = adjacency_matrices[0]
    adj_mat = torch.tensor(adj_mat).to_sparse()
    conv = decagon_pytorch.convolve.SparseDropoutGraphConvActivation(10, 10,
        adj_mat, keep_prob=keep_prob)
    latent = conv(latent)
    return latent


def sparse_dropout_graph_conv_activation_tf(keep_prob=1.):
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    conv_torch = decagon_pytorch.convolve.SparseGraphConv(10, 10,
        torch.tensor(adjacency_matrices[0]).to_sparse())

    weight = tf.constant(conv_torch.weight.detach().numpy())
    nonzero_feat = np.sum(latent > 0)

    latent = dense_to_sparse_tf(latent)
    latent = dropout_sparse_tf(latent, keep_prob,
        nonzero_feat)

    adj_mat = dense_to_sparse_tf(adjacency_matrices[0])

    latent = tf.sparse_tensor_dense_matmul(latent, weight)
    latent = tf.sparse_tensor_dense_matmul(adj_mat, latent)

    latent = tf.nn.relu(latent)

    return latent


def test_sparse_graph_conv():
    latent_torch = sparse_graph_conv_torch()
    latent_tf = sparse_graph_conv_tf()
    assert np.all(latent_torch.detach().numpy() == latent_tf.eval(session = tf.Session()))


def test_sparse_dropout_graph_conv_activation():
    for i in range(11):
        keep_prob = i/10. + np.finfo(np.float32).eps

        latent_torch = sparse_dropout_graph_conv_activation_torch(keep_prob)
        latent_tf = sparse_dropout_graph_conv_activation_tf(keep_prob)

        latent_torch = latent_torch.detach().numpy()
        latent_tf = latent_tf.eval(session = tf.Session())
        print('latent_torch:', latent_torch)
        print('latent_tf:', latent_tf)

        assert np.all(latent_torch - latent_tf < .000001)


def test_sparse_multi_dgca():
    latent_torch = None
    latent_tf = []

    for i in range(11):
        keep_prob = i/10. + np.finfo(np.float32).eps

        latent_torch = sparse_dropout_graph_conv_activation_torch(keep_prob) \
            if latent_torch is None \
            else latent_torch + sparse_dropout_graph_conv_activation_torch(keep_prob)

        latent_tf.append(sparse_dropout_graph_conv_activation_tf(keep_prob))

    latent_torch = torch.nn.functional.normalize(latent_torch, p=2, dim=1)
    latent_tf = tf.add_n(latent_tf)
    latent_tf = tf.nn.l2_normalize(latent_tf, dim=1)

    latent_torch = latent_torch.detach().numpy()
    latent_tf = latent_tf.eval(session = tf.Session())

    assert np.all(latent_torch - latent_tf < .000001)


def test_graph_conv():
    latent_dense = dense_graph_conv_torch()
    latent_sparse = sparse_graph_conv_torch()

    assert np.all(latent_dense.detach().numpy() == latent_sparse.detach().numpy())


# def setup_function(fun):
#     if fun == test_dropout_graph_conv_activation or \
#         fun == test_multi_dgca:
#         print('Disabling dropout for testing...')
#         setup_function.old_dropout = decagon_pytorch.convolve.dropout, \
#             decagon_pytorch.convolve.dropout_sparse
#
#         decagon_pytorch.convolve.dropout = lambda x, keep_prob: x
#         decagon_pytorch.convolve.dropout_sparse = lambda x, keep_prob: x
#
#
# def teardown_function(fun):
#     print('Re-enabling dropout...')
#     if fun == test_dropout_graph_conv_activation or \
#         fun == test_multi_dgca:
#         decagon_pytorch.convolve.dropout, \
#             decagon_pytorch.convolve.dropout_sparse = \
#             setup_function.old_dropout


def flexible_dropout_graph_conv_activation_torch(keep_prob=1.):
    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()
    latent = torch.tensor(latent).to_sparse()
    adj_mat = adjacency_matrices[0]
    adj_mat = torch.tensor(adj_mat).to_sparse()
    conv = decagon_pytorch.convolve.DropoutGraphConvActivation(10, 10,
        adj_mat, keep_prob=keep_prob)
    latent = conv(latent)
    return latent


def _disable_dropout(monkeypatch):
    monkeypatch.setattr(decagon_pytorch.convolve.dense, 'dropout',
        lambda x, keep_prob: x)
    monkeypatch.setattr(decagon_pytorch.convolve.sparse, 'dropout_sparse',
        lambda x, keep_prob: x)
    monkeypatch.setattr(decagon_pytorch.convolve.universal, 'dropout',
        lambda x, keep_prob: x)
    monkeypatch.setattr(decagon_pytorch.convolve.universal, 'dropout_sparse',
        lambda x, keep_prob: x)


def test_dropout_graph_conv_activation(monkeypatch):
    _disable_dropout(monkeypatch)

    for i in range(11):
        keep_prob = i/10.
        if keep_prob == 0:
            keep_prob += np.finfo(np.float32).eps
        print('keep_prob:', keep_prob)

        latent_dense = dense_dropout_graph_conv_activation_torch(keep_prob)
        latent_dense = latent_dense.detach().numpy()
        print('latent_dense:', latent_dense)

        latent_sparse = sparse_dropout_graph_conv_activation_torch(keep_prob)
        latent_sparse = latent_sparse.detach().numpy()
        print('latent_sparse:', latent_sparse)

        latent_flex = flexible_dropout_graph_conv_activation_torch(keep_prob)
        latent_flex = latent_flex.detach().numpy()
        print('latent_flex:', latent_flex)

        nonzero = (latent_dense != 0) & (latent_sparse != 0)

        assert np.all(latent_dense[nonzero] == latent_sparse[nonzero])

        nonzero = (latent_dense != 0) & (latent_flex != 0)

        assert np.all(latent_dense[nonzero] == latent_flex[nonzero])

        nonzero = (latent_sparse != 0) & (latent_flex != 0)

        assert np.all(latent_sparse[nonzero] == latent_flex[nonzero])


def test_multi_dgca(monkeypatch):
    _disable_dropout(monkeypatch)

    keep_prob = .5

    torch.random.manual_seed(0)
    latent, adjacency_matrices = prepare_data()

    latent_sparse = torch.tensor(latent).to_sparse()
    latent = torch.tensor(latent)
    assert np.all(latent_sparse.to_dense().numpy() == latent.numpy())

    adjacency_matrices_sparse = [ torch.tensor(a).to_sparse() for a in adjacency_matrices ]
    adjacency_matrices = [ torch.tensor(a) for a in adjacency_matrices ]

    for i in range(len(adjacency_matrices)):
        assert np.all(adjacency_matrices[i].numpy() == adjacency_matrices_sparse[i].to_dense().numpy())

    torch.random.manual_seed(0)
    multi_sparse = decagon_pytorch.convolve.SparseMultiDGCA([10,] * len(adjacency_matrices), 10, adjacency_matrices_sparse, keep_prob=keep_prob)

    torch.random.manual_seed(0)
    multi = decagon_pytorch.convolve.DenseMultiDGCA([10,] * len(adjacency_matrices), 10, adjacency_matrices, keep_prob=keep_prob)

    print('len(adjacency_matrices):', len(adjacency_matrices))
    print('len(multi_sparse.sparse_dgca):', len(multi_sparse.sparse_dgca))
    print('len(multi.dgca):', len(multi.dgca))

    for i in range(len(adjacency_matrices)):
        assert np.all(multi_sparse.sparse_dgca[i].sparse_graph_conv.weight.detach().numpy() == multi.dgca[i].graph_conv.weight.detach().numpy())

    # torch.random.manual_seed(0)
    latent_sparse = multi_sparse([latent_sparse,] * len(adjacency_matrices))
    # torch.random.manual_seed(0)
    latent = multi([latent,] * len(adjacency_matrices))

    assert np.all(latent_sparse.detach().numpy() == latent.detach().numpy())
