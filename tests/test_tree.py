from binarybeech.tree import Node, Tree
import pandas as pd

def test_node():
    n = Node(value=1.0)
    assert n.is_leaf
    
def test_tree():
    n1 = Node(value=1)
    n2 = Node(value=2)
    n0 = Node(attribute="var",threshold=0.5,branches=[n1,n2], decision_fun=(lambda x,y: x < y))
    t = Tree(root=n0)
    assert t.traverse({"var":0.}).value == 1
    assert t.traverse({"var":1.}).value == 2