import decagon_pytorch.decode.cartesian
import decagon.deep.layers
import numpy as np
import tensorflow as tf
import torch


def _common(decoder_torch, decoder_tf):
    inputs = np.random.rand(20, 10).astype(np.float32)
    inputs_torch = torch.tensor(inputs)
    inputs_tf = {
        0: tf.convert_to_tensor(inputs)
    }
    out_torch = decoder_torch(inputs_torch, inputs_torch)
    out_tf = decoder_tf(inputs_tf)

    assert len(out_torch) == len(out_tf)
    assert len(out_tf) == 7

    for i in range(len(out_torch)):
        assert out_torch[i].shape == out_tf[i].shape

    sess = tf.Session()
    for i in range(len(out_torch)):
        item_torch = out_torch[i].detach().numpy()
        item_tf = out_tf[i].eval(session=sess)
        print('item_torch:', item_torch)
        print('item_tf:', item_tf)
        assert np.all(item_torch - item_tf < .000001)
    sess.close()


def test_dedicom_decoder():
    dedicom_torch = decagon_pytorch.decode.cartesian.DEDICOMDecoder(input_dim=10,
        num_relation_types=7)
    dedicom_tf = decagon.deep.layers.DEDICOMDecoder(input_dim=10, num_types=7,
        edge_type=(0, 0))

    dedicom_tf.vars['global_interaction'] = \
        tf.convert_to_tensor(dedicom_torch.global_interaction.detach().numpy())
    for i in range(dedicom_tf.num_types):
        dedicom_tf.vars['local_variation_%d' % i] = \
            tf.convert_to_tensor(dedicom_torch.local_variation[i].detach().numpy())

    _common(dedicom_torch, dedicom_tf)


def test_dist_mult_decoder():
    distmult_torch = decagon_pytorch.decode.cartesian.DistMultDecoder(input_dim=10,
        num_relation_types=7)
    distmult_tf = decagon.deep.layers.DistMultDecoder(input_dim=10, num_types=7,
        edge_type=(0, 0))

    for i in range(distmult_tf.num_types):
        distmult_tf.vars['relation_%d' % i] = \
            tf.convert_to_tensor(distmult_torch.relation[i].detach().numpy())

    _common(distmult_torch, distmult_tf)


def test_bilinear_decoder():
    bilinear_torch = decagon_pytorch.decode.cartesian.BilinearDecoder(input_dim=10,
        num_relation_types=7)
    bilinear_tf = decagon.deep.layers.BilinearDecoder(input_dim=10, num_types=7,
        edge_type=(0, 0))

    for i in range(bilinear_tf.num_types):
        bilinear_tf.vars['relation_%d' % i] = \
            tf.convert_to_tensor(bilinear_torch.relation[i].detach().numpy())

    _common(bilinear_torch, bilinear_tf)


def test_inner_product_decoder():
    inner_torch = decagon_pytorch.decode.cartesian.InnerProductDecoder(input_dim=10,
        num_relation_types=7)
    inner_tf = decagon.deep.layers.InnerProductDecoder(input_dim=10, num_types=7,
        edge_type=(0, 0))

    _common(inner_torch, inner_tf)
