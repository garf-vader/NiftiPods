import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

import random

import folium

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

G = ox.load_graphml("./graph.graphml")

routes = []
lengths = []
colours = ["blue", "cyan", "green", "black", "magenta", "red", "yellow"]

origs = []
dests = []

print(len(G.edges))

for x in range(200):
    origs.append(random.choice(list(G.nodes)))
    dests.append(random.choice(list(G.nodes)))

# start = time.perf_counter()

for orig, dest in zip(origs, dests):
    try:
        routes.append(nx.shortest_path(G, orig, dest, weight="length"))
        lengths.append(nx.shortest_path_length(G, orig, dest, weight="length"))
    except:
        print(f"Error. No route from {orig} to {dest}.")
        pass

# Extract coordinates of route nodes
route_coordindates = []

for route in routes:
    points = []
    for node_id in route:
        x = G.nodes[node_id]["x"]
        y = G.nodes[node_id]["y"]
        points.append([x, y])
    route_coordindates.append(points)

n_routes = len(route_coordindates)
max_route_len = max([len(x) for x in route_coordindates])

# ox.graph_to_gdfs(route1, nodes=False).explore().save("test.html")

# fig, ax = ox.plot_graph_routes(G, routes)
# route_map.save('test.html')

# Prepare the layout
fig, ax = ox.plot_graph(
    G, node_size=0, edge_linewidth=0.5, show=False, close=False
)  # network
# ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max)) # set the map limits

# Each list is a route
# Length of this list = n_routes
scatter_list = []

# Plot the first scatter plot (starting nodes = initial car locations = hospital locations)
for j in range(n_routes):
    scatter_list.append(
        ax.scatter(
            route_coordindates[j][0][
                0
            ],  # x coordiante of the first node of the j route
            route_coordindates[j][0][
                1
            ],  # y coordiante of the first node of the j route
            label=f"nifti {j}",
            alpha=1,
            s=8,
        )
    )

# plt.legend(frameon=False)


def animate(i):
    # Iterate over all routes = number of cars riding
    for j in range(n_routes):
        # Some routes are shorter than others
        # Therefore we need to use try except with continue construction
        try:
            # Try to plot a scatter plot
            x_j = route_coordindates[j][i][0]
            y_j = route_coordindates[j][i][1]
            scatter_list[j].set_offsets(np.c_[x_j, y_j])
        except:
            # If i became > len(current_route) then continue to the next route
            scatter_list[j].set_offsets(np.c_[0, 0])
            continue


# Make the animation
animation = FuncAnimation(fig, animate, frames=max_route_len)

# plt.show()
# HTML(animation.to_jshtml()) # to display animation in Jupyter Notebook
animation.save("animation.mp4", dpi=300)  # to save animation
