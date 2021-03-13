""" This is just a helper file to figure out this graph package networkx"""


import networkx as nx

G = nx.DiGraph()
nx.set_node_attributes(G, {n: [] for n in G.nodes()}, 'attributes')

# adds nodes, directed edges to graph and then visualizes it
# visualization has a problem with the edge weights
# needs more work

G.add_node(1, value=1)
G.add_node(2, value=2)
G.add_node(3)
G.add_edge(1, 2, weight=2)
G.add_edge(1, 3, weight=3)

node_dict = dict(G.nodes)
edge_dict = dict(G.edges)

G.successors(1) # gives a list of all successor nodes

"""
write function that checks to see if node, action pair is in the graph

check if 1, (1,2) is in the graph
check if 2, (2,1) is in the graph

"""


a=5
pass
# edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
# pos = nx.planar_layout(G)
# nx.draw(G, pos, edgelist=edges)

# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
# nx.draw_networkx(G, arrows=True)

# pp.show()