from graphviz import Digraph

def trace(root):

    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}, filename="DigraphView.svg") # LR -> left to right direction

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape="record") # node for any value
        if n._op: # node for Value obtained from any operation
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the operation node of n2, which is the also the operation node of result of n1 and n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot