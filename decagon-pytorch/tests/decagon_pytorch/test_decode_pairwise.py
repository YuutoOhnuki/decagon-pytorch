import decagon_pytorch.decode.cartesian as cart
import decagon_pytorch.decode.pairwise as pair
import torch


def _common(cart_class, pair_class):
    input_dim = 10
    n_nodes = 20
    num_relation_types = 7

    inputs_row = torch.rand((n_nodes, input_dim))
    inputs_col = torch.rand((n_nodes, input_dim))

    cart_dec = cart_class(input_dim=input_dim,
        num_relation_types=num_relation_types)
    pair_dec = pair_class(input_dim=input_dim,
        num_relation_types=num_relation_types)

    if isinstance(cart_dec, cart.DEDICOMDecoder):
        pair_dec.global_interaction = cart_dec.global_interaction
        pair_dec.local_variation = cart_dec.local_variation
    elif isinstance(cart_dec, cart.InnerProductDecoder):
        pass
    else:
        pair_dec.relation = cart_dec.relation

    cart_pred = cart_dec(inputs_row, inputs_col)
    pair_pred = pair_dec(inputs_row, inputs_col)

    assert isinstance(cart_pred, list)
    assert isinstance(pair_pred, list)

    assert len(cart_pred) == len(pair_pred)
    assert len(cart_pred) == num_relation_types

    for i in range(num_relation_types):
        assert isinstance(cart_pred[i], torch.Tensor)
        assert isinstance(pair_pred[i], torch.Tensor)

        assert cart_pred[i].shape == (n_nodes, n_nodes)
        assert pair_pred[i].shape == (n_nodes,)

        assert torch.all(torch.abs(pair_pred[i] - torch.diag(cart_pred[i])) < 0.000001)


def test_dedicom_decoder():
    _common(cart.DEDICOMDecoder, pair.DEDICOMDecoder)


def test_dist_mult_decoder():
    _common(cart.DistMultDecoder, pair.DistMultDecoder)


def test_bilinear_decoder():
    _common(cart.BilinearDecoder, pair.BilinearDecoder)


def test_inner_product_decoder():
    _common(cart.InnerProductDecoder, pair.InnerProductDecoder)
