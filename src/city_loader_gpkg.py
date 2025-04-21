import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

filepath= "./data/piedmont.gpkg"

# load GeoPackage as node/edge GeoDataFrames indexed as described in OSMnx docs
gdf_nodes = gpd.read_file(filepath, layer='nodes').set_index('osmid')
gdf_edges = gpd.read_file(filepath, layer='edges').set_index(['u', 'v', 'key'])
assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique

# convert the node/edge GeoDataFrames to a MultiDiGraph
graph_attrs = {'crs': 'epsg:4326', 'simplified': True}
G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs)

ox.io.save_graphml(G, filepath="./data/graph_edited.graphml")

print(nx.is_weakly_connected(G))