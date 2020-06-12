from .matrix import NodeType
import torch
from collections import defaultdict


class AdjListRelationType(object):
    def __init__(self, name, node_type_row, node_type_column,
        adjacency_list, adjacency_list_transposed=None):

        #if adjacency_matrix_transposed is not None and \
        #    adjacency_matrix_transposed.shape != adjacency_matrix.transpose(0, 1).shape:
        #    raise ValueError('adjacency_matrix_transposed has incorrect shape')

        self.name = name
        self.node_type_row = node_type_row
        self.node_type_column = node_type_column
        self.adjacency_list = adjacency_list
        self.adjacency_list_transposed = adjacency_list_transposed

    def get_adjacency_list(self, node_type_row, node_type_column):
        if self.node_type_row == node_type_row and \
            self.node_type_column == node_type_column:
            return self.adjacency_list

        elif self.node_type_row == node_type_column and \
            self.node_type_column == node_type_row:
            if self.adjacency_list_transposed is not None:
                return self.adjacency_list_transposed
            else:
                return torch.index_select(self.adjacency_list, 1,
                    torch.LongTensor([1, 0]))

        else:
            raise ValueError('Specified row/column types do not correspond to this relation')


def _verify_adjacency_list(adjacency_list, node_count_row, node_count_col):
    assert isinstance(adjacency_list, torch.Tensor)
    assert len(adjacency_list.shape) == 2
    assert torch.all(adjacency_list[:, 0] >= 0)
    assert torch.all(adjacency_list[:, 0] < node_count_row)
    assert torch.all(adjacency_list[:, 1] >= 0)
    assert torch.all(adjacency_list[:, 1] < node_count_col)


class AdjListData(object):
    def __init__(self):
        self.node_types = []
        self.relation_types = defaultdict(list)

    def add_node_type(self, name, count): # , latent_length):
        self.node_types.append(NodeType(name, count))

    def add_relation_type(self, name, node_type_row, node_type_col, adjacency_list, adjacency_list_transposed=None):
        assert node_type_row >= 0 and node_type_row < len(self.node_types)
        assert node_type_col >= 0 and node_type_col < len(self.node_types)

        node_count_row = self.node_types[node_type_row].count
        node_count_col = self.node_types[node_type_col].count

        _verify_adjacency_list(adjacency_list, node_count_row, node_count_col)
        if adjacency_list_transposed is not None:
            _verify_adjacency_list(adjacency_list_transposed,
                node_count_col, node_count_row)

        self.relation_types[node_type_row, node_type_col].append(
            AdjListRelationType(name, node_type_row, node_type_col,
                adjacency_list, adjacency_list_transposed))
