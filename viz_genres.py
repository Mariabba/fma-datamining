import csv
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

with open("data/genres.csv", "r") as nodecsv:  # Open the file
    nodereader = csv.reader(nodecsv)  # Read the csv
    nodes = [n for n in nodereader][1:]

node_names = [n[0] for n in nodes]  # Get a list of only the node names
edges = [tuple(e[0:3:2]) for e in nodes]  # Retrieve the data
mapping = {x: y for x, _, _, y, _ in nodes}
mapping["0"] = ""

# Build your graph
G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)
G = nx.relabel_nodes(G, mapping)

print(nx.info(G))

# Plot it
nx.draw(G, with_labels=True)
plt.savefig(Path("viz/genres_graph.png"), bbox_inches="tight")
