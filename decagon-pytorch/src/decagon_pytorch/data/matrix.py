#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


from collections import defaultdict
from ..weights import init_glorot


class NodeType(object):
    def __init__(self, name, count):
        self.name = name
        self.count = count


class RelationType(object):
    def __init__(self, name, node_type_row, node_type_column,
        adjacency_matrix, adjacency_matrix_transposed):

        if adjacency_matrix_transposed is not None and \
            adjacency_matrix_transposed.shape != adjacency_matrix.transpose(0, 1).shape:
            raise ValueError('adjacency_matrix_transposed has incorrect shape')

        self.name = name
        self.node_type_row = node_type_row
        self.node_type_column = node_type_column
        self.adjacency_matrix = adjacency_matrix
        self.adjacency_matrix_transposed = adjacency_matrix_transposed

    def get_adjacency_matrix(node_type_row, node_type_column):
        if self.node_type_row == node_type_row and \
            self.node_type_column == node_type_column:
            return self.adjacency_matrix

        elif self.node_type_row == node_type_column and \
            self.node_type_column == node_type_row:
            if self.adjacency_matrix_transposed:
                return self.adjacency_matrix_transposed
            else:
                return self.adjacency_matrix.transpose(0, 1)

        else:
            raise ValueError('Specified row/column types do not correspond to this relation')


class Data(object):
    def __init__(self):
        self.node_types = []
        self.relation_types = defaultdict(list)

    def add_node_type(self, name, count): # , latent_length):
        self.node_types.append(NodeType(name, count))

    def add_relation_type(self, name, node_type_row, node_type_column, adjacency_matrix, adjacency_matrix_transposed=None):
        n = len(self.node_types)
        if node_type_row >= n or node_type_column >= n:
            raise ValueError('Node type index out of bounds, add node type first')
        key = (node_type_row, node_type_column)
        if adjacency_matrix is not None and not adjacency_matrix.is_sparse:
            adjacency_matrix = adjacency_matrix.to_sparse()
        self.relation_types[key].append(RelationType(name, node_type_row, node_type_column, adjacency_matrix, adjacency_matrix_transposed))

    def get_adjacency_matrices(self, node_type_row, node_type_column):
        res = []
        for (i, j), rels in self.relation_types.items():
            if node_type_row not in [i, j] and node_type_column not in [i, j]:
                continue
            for r in rels:
                res.append(r.get_adjacency_matrix(node_type_row, node_type_column))
        return res

    def __repr__(self):
        n = len(self.node_types)
        if n == 0:
            return 'Empty GNN Data'
        s = ''
        s += 'GNN Data with:\n'
        s += '- ' + str(n) + ' node type(s):\n'
        for nt in self.node_types:
            s += '  - ' + nt.name + '\n'
        if len(self.relation_types) == 0:
            s += '- No relation types\n'
            return s.strip()
        n = sum(map(len, self.relation_types))
        s += '- ' + str(n) + ' relation type(s):\n'
        for i in range(n):
            for j in range(n):
                key = (i, j)
                if key not in self.relation_types:
                    continue
                rels = self.relation_types[key]
                s += '  - ' + self.node_types[i].name + ' -- ' + self.node_types[j].name + ':\n'
                for r in rels:
                    s += '    - ' + r.name + '\n'
        return s.strip()
