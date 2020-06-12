#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from collections import defaultdict
from dataclasses import dataclass, field
import torch
from typing import List, \
    Dict, \
    Tuple, \
    Any, \
    Type
from .decode import DEDICOMDecoder, \
    BilinearDecoder


def _equal(x: torch.Tensor, y: torch.Tensor):
    if x.is_sparse ^ y.is_sparse:
        raise ValueError('Cannot mix sparse and dense tensors')

    if not x.is_sparse:
        return (x == y)

    x = x.coalesce()
    indices_x = list(map(tuple, x.indices().transpose(0, 1)))
    order_x = sorted(range(len(indices_x)), key=lambda idx: indices_x[idx])

    y = y.coalesce()
    indices_y = list(map(tuple, y.indices().transpose(0, 1)))
    order_y = sorted(range(len(indices_y)), key=lambda idx: indices_y[idx])

    if not indices_x == indices_y:
        return torch.tensor(0, dtype=torch.uint8)

    return (x.values()[order_x] == y.values()[order_y])


@dataclass
class NodeType(object):
    name: str
    count: int


@dataclass
class RelationTypeBase(object):
    name: str
    node_type_row: int
    node_type_column: int
    adjacency_matrix: torch.Tensor
    adjacency_matrix_backward: torch.Tensor


@dataclass
class RelationType(RelationTypeBase):
    pass


@dataclass
class RelationFamilyBase(object):
    data: 'Data'
    name: str
    node_type_row: int
    node_type_column: int
    is_symmetric: bool
    decoder_class: Type


@dataclass
class RelationFamily(RelationFamilyBase):
    relation_types: List[RelationType] = None

    def __post_init__(self) -> None:
        if not self.is_symmetric and \
            self.decoder_class != DEDICOMDecoder and \
            self.decoder_class != BilinearDecoder:
            raise TypeError('Family is assymetric but the specified decoder_class supports symmetric relations only')

        self.relation_types = []

    def add_relation_type(self,
        name: str, node_type_row: int, node_type_column: int,
        adjacency_matrix: torch.Tensor,
        adjacency_matrix_backward: torch.Tensor = None) -> None:

        name = str(name)
        node_type_row = int(node_type_row)
        node_type_column = int(node_type_column)

        if (node_type_row, node_type_column) != (self.node_type_row, self.node_type_column):
            raise ValueError('Specified node_type_row/node_type_column tuple does not belong to this family')

        if node_type_row < 0 or node_type_row >= len(self.data.node_types):
            raise ValueError('node_type_row outside of the valid range of node types')

        if node_type_column < 0 or node_type_column >= len(self.data.node_types):
            raise ValueError('node_type_column outside of the valid range of node types')

        if adjacency_matrix is None and adjacency_matrix_backward is None:
            raise ValueError('adjacency_matrix and adjacency_matrix_backward cannot both be None')

        if adjacency_matrix is not None and \
            not isinstance(adjacency_matrix, torch.Tensor):
            raise ValueError('adjacency_matrix must be a torch.Tensor')

        # if isinstance(adjacency_matrix_backward, str) and \
        #     adjacency_matrix_backward == 'symmetric':
        #     if self.is_symmetric:
        #         adjacency_matrix_backward = None
        #     else:
        #         adjacency_matrix_backward = adjacency_matrix.transpose(0, 1)

        if adjacency_matrix_backward is not None \
            and not isinstance(adjacency_matrix_backward, torch.Tensor):
            raise ValueError('adjacency_matrix_backward must be a torch.Tensor')

        if adjacency_matrix is not None and \
            adjacency_matrix.shape != (self.data.node_types[node_type_row].count,
                self.data.node_types[node_type_column].count):
            raise ValueError('adjacency_matrix shape must be (num_row_nodes, num_column_nodes)')

        if adjacency_matrix_backward is not None and \
            adjacency_matrix_backward.shape != (self.data.node_types[node_type_column].count,
                self.data.node_types[node_type_row].count):
            raise ValueError('adjacency_matrix_backward shape must be (num_column_nodes, num_row_nodes)')

        if node_type_row == node_type_column and \
            adjacency_matrix_backward is not None:
            raise ValueError('Relation between nodes of the same type must be expressed using a single matrix')

        if self.is_symmetric and adjacency_matrix_backward is not None:
            raise ValueError('Cannot use a custom adjacency_matrix_backward in a symmetric relation family')

        if self.is_symmetric and node_type_row == node_type_column and \
            not torch.all(_equal(adjacency_matrix,
                adjacency_matrix.transpose(0, 1))):
            raise ValueError('Relation family is symmetric but adjacency_matrix is assymetric')

        if self.is_symmetric and node_type_row != node_type_column:
            adjacency_matrix_backward = adjacency_matrix.transpose(0, 1)

        self.relation_types.append(RelationType(name,
            node_type_row, node_type_column,
            adjacency_matrix, adjacency_matrix_backward))

    def node_name(self, index):
        return self.data.node_types[index].name

    def __repr__(self):
        s = 'Relation family %s' % self.name

        for r in self.relation_types:
            s += '\n  - %s%s' % (r.name, ' (two-way)' \
                if (r.adjacency_matrix is not None \
                    and r.adjacency_matrix_backward is not None) \
                        or self.is_symmetric \
                            else '%s <- %s' % (self.node_name(self.node_type_row),
                                self.node_name(self.node_type_column)))

        return s

    def repr_indented(self):
        s = '  - %s' % self.name

        for r in self.relation_types:
            s += '\n    - %s%s' % (r.name, ' (two-way)' \
                if (r.adjacency_matrix is not None \
                    and r.adjacency_matrix_backward is not None) \
                        or self.is_symmetric \
                            else '%s <- %s' % (self.node_name(self.node_type_row),
                                self.node_name(self.node_type_column)))

        return s


class Data(object):
    node_types: List[NodeType]
    relation_families: List[RelationFamily]

    def __init__(self) -> None:
        self.node_types = []
        self.relation_families = []

    def add_node_type(self, name: str, count: int) -> None:
        name = str(name)
        count = int(count)
        if not name:
            raise ValueError('You must provide a non-empty node type name')
        if count <= 0:
            raise ValueError('You must provide a positive node count')
        self.node_types.append(NodeType(name, count))

    def add_relation_family(self, name: str, node_type_row: int,
        node_type_column: int, is_symmetric: bool,
        decoder_class: Type = DEDICOMDecoder):

        fam = RelationFamily(self, name, node_type_row, node_type_column,
            is_symmetric, decoder_class)
        self.relation_families.append(fam)

        return fam

    def __repr__(self):
        n = len(self.node_types)
        if n == 0:
            return 'Empty Icosagon Data'
        s = ''
        s += 'Icosagon Data with:\n'
        s += '- ' + str(n) + ' node type(s):\n'
        for nt in self.node_types:
            s += '  - ' + nt.name + '\n'
        if len(self.relation_families) == 0:
            s += '- No relation families\n'
            return s.strip()

        s += '- %d relation families:\n' % len(self.relation_families)
        for fam in self.relation_families:
            s += fam.repr_indented() + '\n'

        return s.strip()
