import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

G = ox.load_graphml("./data/graph_edited.graphml")


g = ig.Graph.from_networkx(G)
#layout = g.layout(layout='auto')

G = g.to_networkx()

#fig, ax = plt.subplots()
#ig.plot(g, target=ax)

fig, ax = ox.plot_graph(
        G, node_size=0, bgcolor="white", show=False, close=False
    )
plt.show()