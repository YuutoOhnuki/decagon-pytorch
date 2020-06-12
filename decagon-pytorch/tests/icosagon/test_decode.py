from icosagon.decode import DEDICOMDecoder, \
    DistMultDecoder, \
    BilinearDecoder, \
    InnerProductDecoder
import decagon_pytorch.decode.pairwise
import torch


def test_dedicom_decoder_01():
    repr_ = torch.rand(20, 32)
    dec_1 = DEDICOMDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)
    dec_2 = decagon_pytorch.decode.pairwise.DEDICOMDecoder(32, 7, drop_prob=0.,
        activation=torch.sigmoid)
    dec_2.global_interaction = dec_1.global_interaction
    dec_2.local_variation = dec_1.local_variation

    res_1 = [ dec_1(repr_, repr_, k) for k in range(7) ]
    res_2 = dec_2(repr_, repr_)

    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] == res_2[i])


def test_dist_mult_decoder_01():
    repr_ = torch.rand(20, 32)
    dec_1 = DistMultDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)
    dec_2 = decagon_pytorch.decode.pairwise.DistMultDecoder(32, 7, drop_prob=0.,
        activation=torch.sigmoid)
    dec_2.relation = dec_1.relation

    res_1 = [ dec_1(repr_, repr_, k) for k in range(7) ]
    res_2 = dec_2(repr_, repr_)

    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] == res_2[i])


def test_bilinear_decoder_01():
    repr_ = torch.rand(20, 32)
    dec_1 = BilinearDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)
    dec_2 = decagon_pytorch.decode.pairwise.BilinearDecoder(32, 7, drop_prob=0.,
        activation=torch.sigmoid)
    dec_2.relation = dec_1.relation

    res_1 = [ dec_1(repr_, repr_, k) for k in range(7) ]
    res_2 = dec_2(repr_, repr_)

    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] == res_2[i])


def test_inner_product_decoder_01():
    repr_ = torch.rand(20, 32)
    dec_1 = InnerProductDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)
    dec_2 = decagon_pytorch.decode.pairwise.InnerProductDecoder(32, 7, drop_prob=0.,
        activation=torch.sigmoid)

    res_1 = [ dec_1(repr_, repr_, k) for k in range(7) ]
    res_2 = dec_2(repr_, repr_)

    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] == res_2[i])


def test_is_dedicom_not_symmetric_01():
    repr_1 = torch.rand(20, 32)
    repr_2 = torch.rand(20, 32)
    dec = DEDICOMDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)

    res_1 = [ dec(repr_1, repr_2, k) for k in range(7) ]
    res_2 = [ dec(repr_2, repr_1, k) for k in range(7) ]


    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert not torch.all(res_1[i] - res_2[i] < 0.000001)


def test_is_dist_mult_symmetric_01():
    repr_1 = torch.rand(20, 32)
    repr_2 = torch.rand(20, 32)
    dec = DistMultDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)

    res_1 = [ dec(repr_1, repr_2, k) for k in range(7) ]
    res_2 = [ dec(repr_2, repr_1, k) for k in range(7) ]


    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] - res_2[i] < 0.000001)


def test_is_bilinear_not_symmetric_01():
    repr_1 = torch.rand(20, 32)
    repr_2 = torch.rand(20, 32)
    dec = BilinearDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)

    res_1 = [ dec(repr_1, repr_2, k) for k in range(7) ]
    res_2 = [ dec(repr_2, repr_1, k) for k in range(7) ]

    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert not torch.all(res_1[i] - res_2[i] < 0.000001)


def test_is_inner_product_symmetric_01():
    repr_1 = torch.rand(20, 32)
    repr_2 = torch.rand(20, 32)
    dec = InnerProductDecoder(32, 7, keep_prob=1.,
        activation=torch.sigmoid)

    res_1 = [ dec(repr_1, repr_2, k) for k in range(7) ]
    res_2 = [ dec(repr_2, repr_1, k) for k in range(7) ]


    assert isinstance(res_1, list)
    assert isinstance(res_2, list)

    assert len(res_1) == len(res_2)

    for i in range(len(res_1)):
        assert torch.all(res_1[i] - res_2[i] < 0.000001)
