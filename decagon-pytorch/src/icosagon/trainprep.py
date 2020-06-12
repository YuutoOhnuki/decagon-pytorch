#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from .sampling import fixed_unigram_candidate_sampler
import torch
from dataclasses import dataclass, \
    field
from typing import Any, \
    List, \
    Tuple, \
    Dict
from .data import NodeType, \
    RelationType, \
    RelationTypeBase, \
    RelationFamily, \
    RelationFamilyBase, \
    Data
from collections import defaultdict
from .normalize import norm_adj_mat_one_node_type, \
    norm_adj_mat_two_node_types
import numpy as np


@dataclass
class TrainValTest(object):
    train: Any
    val: Any
    test: Any


@dataclass
class PreparedRelationType(RelationTypeBase):
    edges_pos: TrainValTest
    edges_neg: TrainValTest


@dataclass
class PreparedRelationFamily(RelationFamilyBase):
    relation_types: List[PreparedRelationType]


@dataclass
class PreparedData(object):
    node_types: List[NodeType]
    relation_families: List[PreparedRelationFamily]


def train_val_test_split_edges(edges: torch.Tensor,
    ratios: TrainValTest) -> TrainValTest:

    if not isinstance(edges, torch.Tensor):
        raise ValueError('edges must be a torch.Tensor')

    if len(edges.shape) != 2 or edges.shape[1] != 2:
        raise ValueError('edges shape must be (num_edges, 2)')

    if not isinstance(ratios, TrainValTest):
        raise ValueError('ratios must be a TrainValTest')

    if ratios.train + ratios.val + ratios.test != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')

    order = torch.randperm(len(edges))
    edges = edges[order, :]
    n = round(len(edges) * ratios.train)
    edges_train = edges[:n]
    n_1 = round(len(edges) * (ratios.train + ratios.val))
    edges_val = edges[n:n_1]
    edges_test = edges[n_1:]

    return TrainValTest(edges_train, edges_val, edges_test)


def get_edges_and_degrees(adj_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if adj_mat.is_sparse:
        adj_mat = adj_mat.coalesce()
        degrees = torch.zeros(adj_mat.shape[1], dtype=torch.int64)
        degrees = degrees.index_add(0, adj_mat.indices()[1],
            torch.ones(adj_mat.indices().shape[1], dtype=torch.int64))
        edges_pos = adj_mat.indices().transpose(0, 1)
    else:
        degrees = adj_mat.sum(0)
        edges_pos = torch.nonzero(adj_mat)
    return edges_pos, degrees


def prepare_adj_mat(adj_mat: torch.Tensor,
    ratios: TrainValTest) -> Tuple[TrainValTest, TrainValTest]:

    if not isinstance(adj_mat, torch.Tensor):
        raise ValueError('adj_mat must be a torch.Tensor')

    edges_pos, degrees = get_edges_and_degrees(adj_mat)

    neg_neighbors = fixed_unigram_candidate_sampler(
        edges_pos[:, 1].view(-1, 1), degrees, 0.75)
    print(edges_pos.dtype)
    print(neg_neighbors.dtype)
    edges_neg = torch.cat((edges_pos[:, 0].view(-1, 1), neg_neighbors.view(-1, 1)), 1)

    edges_pos = train_val_test_split_edges(edges_pos, ratios)
    edges_neg = train_val_test_split_edges(edges_neg, ratios)

    adj_mat_train = torch.sparse_coo_tensor(indices = edges_pos.train.transpose(0, 1),
        values=torch.ones(len(edges_pos.train), dtype=adj_mat.dtype))

    return adj_mat_train, edges_pos, edges_neg


def prep_rel_one_node_type(r: RelationType,
    ratios: TrainValTest) -> PreparedRelationType:

    adj_mat = r.adjacency_matrix
    adj_mat_train, edges_pos, edges_neg = prepare_adj_mat(adj_mat, ratios)

    print('adj_mat_train:', adj_mat_train)
    adj_mat_train = norm_adj_mat_one_node_type(adj_mat_train)

    return PreparedRelationType(r.name, r.node_type_row, r.node_type_column,
        adj_mat_train, None, edges_pos, edges_neg)


def prep_rel_two_node_types_sym(r: RelationType,
    ratios: TrainValTest) -> PreparedRelationType:

    adj_mat = r.adjacency_matrix
    adj_mat_train, edges_pos, edges_neg = prepare_adj_mat(adj_mat, ratios)

    return PreparedRelationType(r.name, r.node_type_row,
        r.node_type_column,
        norm_adj_mat_two_node_types(adj_mat_train),
        norm_adj_mat_two_node_types(adj_mat_train.transpose(0, 1)),
        edges_pos, edges_neg)


def prep_rel_two_node_types_asym(r: RelationType,
    ratios: TrainValTest) -> PreparedRelationType:

    if r.adjacency_matrix is not None:
        adj_mat_train, edges_pos, edges_neg =\
            prepare_adj_mat(r.adjacency_matrix, ratios)
    else:
        adj_mat_train, edges_pos, edges_neg = \
            None, torch.zeros((0, 2)), torch.zeros((0, 2))

    if r.adjacency_matrix_backward is not None:
        adj_mat_back_train, edges_back_pos, edges_back_neg = \
            prepare_adj_mat(r.adjacency_matrix_backward, ratios)
    else:
        adj_mat_back_train, edges_back_pos, edges_back_neg = \
            None, torch.zeros((0, 2)), torch.zeros((0, 2))

    edges_pos = torch.cat((edges_pos, edges_back_pos), dim=0)
    edges_neg = torch.cat((edges_neg, edges_back_neg), dim=0)

    return PreparedRelationType(r.name, r.node_type_row,
        r.node_type_column,
        norm_adj_mat_two_node_types(adj_mat_train),
        norm_adj_mat_two_node_types(adj_mat_back_train),
        edges_pos, edges_neg)


def prepare_relation_type(r: RelationType,
    ratios: TrainValTest, is_symmetric: bool) -> PreparedRelationType:

    if not isinstance(r, RelationType):
        raise ValueError('r must be a RelationType')

    if not isinstance(ratios, TrainValTest):
        raise ValueError('ratios must be a TrainValTest')

    if r.node_type_row == r.node_type_column:
        return prep_rel_one_node_type(r, ratios)
    elif is_symmetric:
        return prep_rel_two_node_types_sym(r, ratios)
    else:
        return prep_rel_two_node_types_asym(r, ratios)


def prepare_relation_family(fam: RelationFamily) -> PreparedRelationFamily:
    relation_types = []

    for r in fam.relation_types:
        relation_types.append(prepare_relation_type(r, ratios, fam.is_symmetric))

    return PreparedRelationFamily(fam.data, fam.name,
        fam.node_type_row, fam.node_type_column,
        fam.is_symmetric, fam.decoder_class,
        relation_types)


def prepare_training(data: Data, ratios: TrainValTest) -> PreparedData:
    if not isinstance(data, Data):
        raise ValueError('data must be of class Data')

    relation_families = [ prepare_relation_family(fam) \
        for fam in data.relation_families ]

    return PreparedData(data.node_types, relation_families)
