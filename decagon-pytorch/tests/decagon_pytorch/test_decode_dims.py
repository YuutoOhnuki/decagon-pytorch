from decagon_pytorch.decode.cartesian import DEDICOMDecoder, \
    DistMultDecoder, \
    BilinearDecoder, \
    InnerProductDecoder
import torch


def _common(decoder_class):
    decoder = decoder_class(input_dim=10, num_relation_types=1)
    inputs = torch.rand((20, 10))
    pred = decoder(inputs, inputs)

    assert isinstance(pred, list)
    assert len(pred) == 1

    assert isinstance(pred[0], torch.Tensor)
    assert pred[0].shape == (20, 20)



def test_dedicom_decoder():
    _common(DEDICOMDecoder)


def test_dist_mult_decoder():
    _common(DistMultDecoder)


def test_bilinear_decoder():
    _common(BilinearDecoder)


def test_inner_product_decoder():
    _common(InnerProductDecoder)


def test_batch_matrix_multiplication():
    input_dim = 10
    inputs = torch.rand((20, 10))

    decoder = DEDICOMDecoder(input_dim=input_dim, num_relation_types=1)
    out_dec = decoder(inputs, inputs)

    relation = decoder.local_variation[0]
    global_interaction = decoder.global_interaction
    act = decoder.activation
    relation = torch.diag(relation)
    product1 = torch.mm(inputs, relation)
    product2 = torch.mm(product1, global_interaction)
    product3 = torch.mm(product2, relation)
    rec = torch.mm(product3, torch.transpose(inputs, 0, 1))
    rec = act(rec)

    print('rec:', rec)
    print('out_dec:', out_dec)

    assert torch.all(rec == out_dec[0])


def test_single_prediction_01():
    input_dim = 10
    inputs = torch.rand((20, 10))

    decoder = DEDICOMDecoder(input_dim=input_dim, num_relation_types=1)
    dec_all = decoder(inputs, inputs)
    dec_one = decoder(inputs[0:1], inputs[0:1])

    assert torch.abs(dec_all[0][0, 0] - dec_one[0][0, 0]) < 0.000001


def test_single_prediction_02():
    input_dim = 10
    inputs = torch.rand((20, 10))

    decoder = DEDICOMDecoder(input_dim=input_dim, num_relation_types=1)
    dec_all = decoder(inputs, inputs)
    dec_one = decoder(inputs[0:1], inputs[1:2])

    assert torch.abs(dec_all[0][0, 1] - dec_one[0][0, 0]) < 0.000001
    assert dec_one[0].shape == (1, 1)


def test_pairwise_prediction():
    n_nodes = 20
    input_dim = 10
    inputs_row = torch.rand((n_nodes, input_dim))
    inputs_col = torch.rand((n_nodes, input_dim))

    decoder = DEDICOMDecoder(input_dim=input_dim, num_relation_types=1)
    dec_all = decoder(inputs_row, inputs_col)

    relation = torch.diag(decoder.local_variation[0])
    global_interaction = decoder.global_interaction
    act = decoder.activation
    product1 = torch.mm(inputs_row, relation)
    product2 = torch.mm(product1, global_interaction)
    product3 = torch.mm(product2, relation)
    rec = torch.bmm(product3.view(product3.shape[0], 1, product3.shape[1]),
        inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))

    assert rec.shape == (n_nodes, 1, 1)

    rec = torch.flatten(rec)
    rec = act(rec)

    assert torch.all(torch.abs(rec - torch.diag(dec_all[0])) < 0.000001)
