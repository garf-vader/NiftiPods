import osmnx as ox
import networkx as nx
import copy

import random
import time
import numpy as np

import cProfile

import custom_classes as cc

import pickle
import itertools

def reset_reservations(graph):
    nx.set_node_attributes(
        graph, np.zeros(0), name="reservations"
        )  # creates "empty" attribute for assigning reservations

graph_name = "piedmont_clean"
G = ox.load_graphml(f"./cleaner_code/{graph_name}.graphml")

reset_reservations(G)

print(f"Number of nodes: {len(G.nodes)}")

master_list = [
    53040794,
    53028190,
    246744892,
    256338428,
    53035699,
    53149343,
    53146451,
    53092226,
    53086074,
    53065785,
    3905291938,
]  # source nodes for simplicity

#### I have realised that due to platooning we wont have any issue of collisions on nodes with only 2 degrees (middle of a road nodes)
for node in master_list:
    G.nodes[node]["street_count"] = 2
#### Therefore I have created a simple code that will crudely solve this issue for now

count = itertools.count()

def generate_data(N, dt, wait_runs, reroute_runs):
    for n1 in range(wait_runs):
        start1 = time.perf_counter()
        method = "wait_only"
        pods = cc.fleet(G, N, master_list, dt, method)
        pods.generate_routes()
        reset_reservations(G)
        filename = f"{method}-N-{N}-dt-{dt:.2f}-{graph_name}-{next(count):02d}"
        with open(f"./fleets/{filename}.pickle", "wb") as f:
            pickle.dump(pods, f)
        print(f"Time taken: {time.perf_counter() - start1}")
    for n2 in range(reroute_runs):
        start1 = time.perf_counter()
        method = "wait_and_reroute"
        pods = cc.fleet(G, N, master_list, dt, method)
        pods.generate_routes()
        reset_reservations(G)
        filename = f"{method}-N-{N}-dt-{dt:.2f}-{graph_name}-{next(count):02d}"
        with open(f"./fleets/{filename}.pickle", "wb") as f:
            pickle.dump(pods, f)
        print(f"Time taken: {time.perf_counter() - start1}")
    if wait_runs+reroute_runs == 0:
        import matplotlib.pyplot as plt
        pods = cc.fleet(G, N, master_list, dt, "wait_only")
        pods.generate_routes()
        reset_reservations(G)
        fig, ax = ox.plot_graph_routes(G, pods.routes, bgcolor = "white", route_linewidth=6, node_size=0)
        plt.show()

for each in range(3,20):
    generate_data(N = each, dt = 2, wait_runs = 2, reroute_runs = 2)

#t = 0  # initial time
#N = 100  # number of pods
#N = int(3600/dt)
#N = 36000
#dt = 2  # period of pod production


"""for node in master_list: # annotates the node IDs
    text = node
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]
    ax.annotate(text, (x, y), c='y')"""