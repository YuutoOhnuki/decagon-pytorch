from decagon_pytorch.data import AdjListData, \
    AdjListRelationType
import torch
import pytest


def _get_list():
    lst = torch.tensor([
        [0, 1],
        [0, 3],
        [0, 5],
        [0, 7]
    ])
    return lst


def test_adj_list_relation_type_01():
    lst = _get_list()
    rel = AdjListRelationType('Test', 0, 0, lst)
    assert torch.all(rel.get_adjacency_list(0, 0) == lst)


def test_adj_list_relation_type_02():
    lst = _get_list()
    rel = AdjListRelationType('Test', 0, 1, lst)
    assert torch.all(rel.get_adjacency_list(0, 1) == lst)
    lst_2 = torch.tensor([
        [1, 0],
        [3, 0],
        [5, 0],
        [7, 0]
    ])
    assert torch.all(rel.get_adjacency_list(1, 0) == lst_2)


def test_adj_list_relation_type_03():
    lst = _get_list()
    lst_2 = torch.tensor([
        [2, 0],
        [4, 0],
        [6, 0],
        [8, 0]
    ])
    rel = AdjListRelationType('Test', 0, 1, lst, lst_2)
    assert torch.all(rel.get_adjacency_list(0, 1) == lst)
    assert torch.all(rel.get_adjacency_list(1, 0) == lst_2)


def test_adj_list_data_01():
    lst = _get_list()
    d = AdjListData()
    with pytest.raises(AssertionError):
        d.add_relation_type('Test', 0, 1, lst)
    d.add_node_type('Drugs', 5)
    with pytest.raises(AssertionError):
        d.add_relation_type('Test', 0, 0, lst)
    d = AdjListData()
    d.add_node_type('Drugs', 8)
    d.add_relation_type('Test', 0, 0, lst)


def test_adj_list_data_02():
    lst = _get_list()
    d = AdjListData()
    d.add_node_type('Drugs', 10)
    d.add_node_type('Proteins', 10)
    d.add_relation_type('Target', 0, 1, lst)
