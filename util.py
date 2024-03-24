import networkx as nx
from deap import gp
from matplotlib import pyplot as plt


def draw_individual(individual, pset):
    individual = gp.PrimitiveTree.from_string(individual, pset)
    print("Final individual " + str(individual))
    nodes, edges, labels = gp.graph(individual)
    print("Graph details: " + str(nodes) + "\n" + str(edges) + "\n" + str(labels))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos, node_size=600, node_color='skyblue')
    nx.draw_networkx_edges(g, pos)

    # Replace primitive symbols with their respective symbols in the labels
    for i, label in enumerate(labels.values()):
        if label == 'mul':
            labels[i] = '*'
        elif label == 'protected_add':
            labels[i] = '+'
        elif label == 'sub':
            labels[i] = '–'
    nx.draw_networkx_labels(g, pos, labels, font_size=14, horizontalalignment='center', verticalalignment='center')
    # Removing the border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.show()
