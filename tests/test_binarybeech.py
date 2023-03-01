from binarybeech.binarybeech import *


def test_node():
    n = Node(value=1.0)
    assert n.is_leaf
