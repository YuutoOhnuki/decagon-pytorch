from .sampling import fixed_unigram_candidate_sampler
import torch


def train_val_test_split_edges(edges, ratios):
    train_ratio, val_ratio, test_ratio = ratios

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError('Train, validation and test ratios must add up to 1')

    order = torch.randperm(len(edges))
    edges = edges[order, :]
    n = round(len(edges) * train_ratio)
    edges_train = edges[:n]
    n_1 = round(len(edges) * (train_ratio + val_ratio))
    edges_val = edges[n:n_1]
    edges_test = edges[n_1:]

    return edges_train, edges_val, edges_test


def prepare_adj_mat(adj_mat, ratios):
    degrees = adj_mat.sum(0)
    edges_pos = torch.nonzero(adj_mat)

    neg_neighbors = fixed_unigram_candidate_sampler(edges_pos[:, 1],
        len(edges), degrees, 0.75)
    edges_neg = torch.cat((edges_pos[:, 0], neg_neighbors.view(-1, 1)), 1)

    edges_pos = (edges_pos_train, edges_pos_val, edges_pos_test) = \
        train_val_test_split_edges(edges_pos, ratios)
    edges_neg = (edges_neg_train, edges_neg_val, edges_neg_test) = \
        train_val_test_split_edges(edges_neg, ratios)

    return edges_pos, edges_neg


class PreparedRelation(object):
    def __init__(self, node_type_row, node_type_column,
        adj_mat_train, adj_mat_train_trans,
        edges_pos, edges_neg, edges_pos_trans, edges_neg_trans):

        self.adj_mat_train = adj_mat_train
        self.adj_mat_train_trans = adj_mat_train_trans
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg
        self.edges_pos_trans = edges_pos_trans
        self.edges_neg_trans = edges_neg_trans


def prepare_relation(r, ratios):
    adj_mat = r.get_adjacency_matrix(r.node_type_row, r.node_type_column)
    edges_pos, edges_neg = prepare_adj_mat(adj_mat)

    # adj_mat_train = torch.zeros_like(adj_mat)
    # adj_mat_train[edges_pos[0][:, 0], edges_pos[0][:, 0]] = 1
    adj_mat_train = torch.sparse_coo_tensor(indices = edges_pos[0].transpose(0, 1),
        values=torch.ones(len(edges_pos[0]), dtype=adj_mat.dtype))

    if r.node_type_row != r.node_type_col:
        adj_mat_trans = r.get_adjacency_matrix(r.node_type_col, r.node_type_row)
        edges_pos_trans, edges_neg_trans = prepare_adj_mat(adj_mat_trans)
        adj_mat_train_trans = torch.sparse_coo_tensor(indices = edges_pos_trans[0].transpose(0, 1),
            values=torch.ones(len(edges_pos_trans[0]), dtype=adj_mat_trans.dtype))
    else:
        adj_mat_train_trans = adj_mat_trans = \
            edge_pos_trans = edge_neg_trans = None

    return PreparedRelation(r.node_type_row, r.node_type_column,
        adj_mat_train, adj_mat_trans_train,
        edges_pos, edges_neg, edges_pos_trans, edges_neg_trans)


def prepare_training(data):
    for (node_type_row, node_type_column), rels in data.relation_types:
        for r in rels:
            prep_relation_edges()
