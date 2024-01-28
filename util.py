import networkx as nx
from deap import gp
from matplotlib import pyplot as plt


def draw_individual(individual):
    nodes, edges, labels = gp.graph(individual)
    print("Graph details: " + str(nodes) + "\n" + str(edges) + "\n" + str(labels))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
