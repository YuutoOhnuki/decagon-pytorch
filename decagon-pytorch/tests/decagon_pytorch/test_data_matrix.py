from decagon_pytorch.data import Data


def test_data():
    d = Data()
    d.add_node_type('Gene', 1000)
    d.add_node_type('Drug', 100)
    d.add_relation_type('Target', 1, 0, None)
    d.add_relation_type('Interaction', 0, 0, None)
    d.add_relation_type('Side Effect: Nausea', 1, 1, None)
    d.add_relation_type('Side Effect: Infertility', 1, 1, None)
    d.add_relation_type('Side Effect: Death', 1, 1, None)
    print(d)
