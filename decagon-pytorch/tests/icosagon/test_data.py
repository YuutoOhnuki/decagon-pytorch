#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from icosagon import Data
import torch
import pytest


def test_data_01():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    dummy_0 = torch.zeros((100, 1000))
    dummy_1 = torch.zeros((1000, 100))
    dummy_2 = torch.zeros((100, 100))
    dummy_3 = torch.zeros((1000, 1000))
    fam = d.add_relation_family('Drug-Gene', 1, 0, True)
    fam.add_relation_type('Target', 1, 0, dummy_0)
    fam = d.add_relation_family('Gene-Gene', 0, 0, True)
    fam.add_relation_type('Interaction', 0, 0, dummy_3)
    fam = d.add_relation_family('Drug-Drug', 1, 1, True)
    fam.add_relation_type('Side Effect: Nausea', 1, 1, dummy_2)
    fam.add_relation_type('Side Effect: Infertility', 1, 1, dummy_2)
    fam.add_relation_type('Side Effect: Death', 1, 1, dummy_2)
    print(d)


def test_data_02():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)

    dummy_0 = torch.zeros((100, 1000))
    dummy_1 = torch.zeros((1000, 100))
    dummy_2 = torch.zeros((100, 100))
    dummy_3 = torch.zeros((1000, 1000))

    fam = d.add_relation_family('Drug-Gene', 1, 0, True)
    with pytest.raises(ValueError):
        fam.add_relation_type('Target', 1, 0, dummy_1)

    fam = d.add_relation_family('Gene-Gene', 0, 0, True)
    with pytest.raises(ValueError):
        fam.add_relation_type('Interaction', 0, 0, dummy_2)

    fam = d.add_relation_family('Drug-Drug', 1, 1, True)
    with pytest.raises(ValueError):
        fam.add_relation_type('Side Effect: Nausea', 1, 1, dummy_3)
    with pytest.raises(ValueError):
        fam.add_relation_type('Side Effect: Infertility', 1, 1, dummy_3)
    with pytest.raises(ValueError):
        fam.add_relation_type('Side Effect: Death', 1, 1, dummy_3)
    print(d)


def test_data_03():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    fam = d.add_relation_family('Drug-Gene', 1, 0, True)
    with pytest.raises(ValueError):
        fam.add_relation_type('Target', 1, 0, None)
    print(d)
