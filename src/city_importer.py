import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Specify the name that is used to seach for the data
place_name = 'Piedmont, CA, USA'

# Fetch OSM street network from the location
G = ox.graph_from_place(place_name, network_type='drive')#, simplify=False)

print(nx.is_strongly_connected(G))

G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)

D = ox.convert.to_digraph(G, weight="travel_time")

# save graph as a geopackage or graphml file
ox.io.save_graphml(G, filepath="./data/graph.graphml")
ox.save_graph_geopackage(G, filepath="./data/piedmont.gpkg")