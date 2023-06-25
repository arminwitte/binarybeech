from binarybeech.tree import Node, Tree


def test_node():
    n = Node(value=1.0)
    assert n.is_leaf


def test_tree():
    n1 = Node(value=1)
    n2 = Node(value=2)
    n0 = Node(
        attribute="var",
        threshold=0.5,
        branches=[n1, n2],
        decision_fun=(lambda x, y: x < y),
    )
    t = Tree(root=n0)
    assert t.traverse({"var": 0.0}).value == 1
    assert t.traverse({"var": 1.0}).value == 2

def test_tree_parent():
    n1 = Node(value=1)
    n2 = Node(value=2)
    n0 = Node(
        attribute="var",
        threshold=0.5,
        branches=[n1, n2],
        decision_fun=(lambda x, y: x < y),
    )
    n1.parent = n0
    n2.parent = n0
    t = Tree(root=n0)
    assert isinstance(t.traverse({"var": 0.0}).parent, Node)
    assert len(t.leafs()) == 2
    
def test_tree_to_json():
    def decfun(x, y):
        return x < y
    n1 = Node(value=1)
    n2 = Node(value=2)
    n0 = Node(
        attribute="var",
        threshold=0.5,
        branches=[n1, n2],
        decision_fun = decfun,
    )
    t = Tree(root=n0)
    
    assert isinstance(t.to_json(),str)

