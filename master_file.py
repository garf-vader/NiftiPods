import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import copy

import random
import time
import pandas as pd
import numpy as np
import geopandas as gpd

import cProfile

import concurrent.futures
#import graphblas_algorithms as ga

def collision_checker(gap_array, junction_time):
    return np.any(
        (gap_array < (junction_time / 2)) & (gap_array > -(junction_time / 2))
    )


def gapsolver(X, gap_array, junction_time):
    gap_trial_array = gap_array + X
    result = collision_checker(gap_trial_array, junction_time)
    # print(f"X: {X}, Gap array: {gap_trial_array}, Result: {result}")
    return result


def find_positive_x(gap_array, junction_time, tolerance=1e-2):
    lower_bound, upper_bound = 0.0, 1.0
    while gapsolver(upper_bound, gap_array, junction_time):
        lower_bound, upper_bound = upper_bound, upper_bound * 2.0

    while upper_bound - lower_bound > tolerance:
        mid_point = (upper_bound + lower_bound) / 2.0
        if gapsolver(mid_point, gap_array, junction_time):
            lower_bound = mid_point
        else:
            upper_bound = mid_point

    return upper_bound


class nifti:
    def __init__(self, name, orig, dest, graph, t):
        self.name = name
        self.orig = orig
        self.dest = dest
        self.graph = graph
        self.t = t
        self.speed = 4
        # temp values for comparison
        self.route = []
        self.timetable = []
        self.wait_time = np.inf
        self.triptime = np.inf

    def compute_route(self, temp_route):
        route_roadlengths = [
            self.graph.edges[x, y, 0]["length"] / self.speed
            for x, y in zip(temp_route, temp_route[1:])
        ]
        route_roadlengths.insert(0, 0)

        jt = 2  # time to pass through junction

        temp_timetable = np.cumsum(route_roadlengths) + self.t
        temp_triptime = (temp_timetable[-1] - temp_timetable[0]) / self.speed  # in seconds

        # numpy timetable method

        gaps = []
        for i, node in enumerate(temp_route):
            gaps.append(temp_timetable[i] - self.graph.nodes[node]["reservations"])

        gaps = np.concatenate(gaps)  # returns array of values
        # representing the time gap between the pod and other
        # pods at each junction (a positive value means this pod will get there after the other pod)

        if collision_checker(gaps, jt):
            temp_wait_time = find_positive_x(gaps, jt)

        else:
            temp_wait_time = 0

        temp_timetable = temp_timetable + temp_wait_time

        return temp_route, temp_timetable, temp_wait_time, temp_triptime
    
    def dist(self, a, b): # heuristic for astar, didnt really help, probably too few nodes
        x1, y1 = self.graph.nodes[a]["x"], self.graph.nodes[a]["y"]
        x2, y2 = self.graph.nodes[b]["x"], self.graph.nodes[b]["y"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def routeplan_wait_only(self):
        temp_route = nx.astar_path( G=self.graph, source=self.orig, target=self.dest, heuristic=self.dist, weight="length")

        self.route, self.timetable, self.wait_time, self.triptime = self.compute_route(temp_route)

        for i, node in enumerate(self.route):
            if (
                self.graph.nodes[node]["street_count"] > 2
            ):  # prevents seemless nodes causing slowing
                self.graph.nodes[node]["reservations"] = np.append(
                    self.graph.nodes[node]["reservations"], self.timetable[i]
                )

        self.compute_time = time.perf_counter() - start

    def routeplan_wait_and_reroute(self):
        routes = ox.routing.k_shortest_paths(
            self.graph, self.orig, self.dest, k=4, weight="length"
        )

        results = []
        for temp_route in routes:
            results.append(self.compute_route(temp_route))

        best_result = min(results, key=lambda result: result[2])

        self.route, self.timetable, self.wait_time, self.triptime = best_result

        for i, node in enumerate(self.route):
            if (
                self.graph.nodes[node]["street_count"] > 2
            ):  # prevents seemless nodes causing slowing
                self.graph.nodes[node]["reservations"] = np.append(
                    self.graph.nodes[node]["reservations"], self.timetable[i]
                )
        self.compute_time = time.perf_counter() - start

    def compute_route_parallel(self): # didnt really work
        start = time.perf_counter()

        routes = ox.routing.k_shortest_paths(
            self.graph, self.orig, self.dest, k=4, weight="length"
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.compute_route, temp_route) for temp_route in routes]

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

            # Retrieve results
            results = [future.result() for future in futures]

        # Find the best result
        best_result = min(results, key=lambda result: result[2])

        self.route, self.timetable, self.wait_time, self.triptime = best_result

        #print(f"Pod {self.name} total: {time.perf_counter() - start}")

        for i, node in enumerate(self.route):
                if (
                    self.graph.nodes[node]["street_count"] > 2
                ):  # prevents seemless nodes causing slowing
                    self.graph.nodes[node]["reservations"] = np.append(
                        self.graph.nodes[node]["reservations"], self.timetable[i]
                )
                    
        self.compute_time = time.perf_counter() - start

def reset_reservations(graph):
    nx.set_node_attributes(
        graph, np.zeros(0), name="reservations"
        )  # creates "empty" attribute for assigning reservations
    
def initialise_pods(graph, master_list, N, dt):
    t = 0
    pods = []
    for pod_number in range(N):
        orig, dest = random.sample(master_list, 2)
        pods.append(nifti(pod_number, orig, dest, graph, t))
        t += dt
    return pods

def generate_routes(pods, method):
    start = time.perf_counter()
    routes = []
    for pod in pods:
        if method == "wait_only":
            pod.routeplan_wait_only()
        if method == "wait_and_reroute":
                pod.routeplan_wait_and_reroute()
        routes.append(pod.route)
    print(time.perf_counter() - start)
    return routes

G = ox.load_graphml("./data/graph_edited.graphml")

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
#### Therefore I have created a simple code that will crudely solve this issue for now

for node in master_list:
    G.nodes[node]["street_count"] = 2

start = time.perf_counter()

wait_plot_frequency = []
wait_plot1 = []
wait_plot2 = []

for dt in [1, 2, 3, 4]:#,10, 100, 1000]:

    t = 0  # initial time
    N = 100  # number of pods
    #N = int(3600/dt)
    #dt = 2  # period of pod production

    pods_wait = initialise_pods(G, master_list, N, dt)

    pods_wait_and_reroute = copy.deepcopy(pods_wait)

    pr = cProfile.Profile()
    pr.enable()

    routes_wait = generate_routes(pods_wait, "wait_only")
    reset_reservations(G)
    #routes_wait_and_reroute = generate_routes(pods2, "wait_and_reroute")


    pr.disable()
    #pr.print_stats(sort='cumulative')


    wait_plot_frequency.append(3600/dt)
    wait_plot1.append(np.mean(np.array([pod.wait_time for pod in pods_wait])))
    #wait_plot2.append(np.mean(np.array([pod.wait_time for pod in pods_wait_and_reroute])))

    """wait_times = np.array([pod.wait_time for pod in pods_wait])
    print(f"Average wait time was {np.mean(wait_times)} for pod1 N = {N}")
    wait_times2 = np.array([pod.wait_time for pod in pods_wait_and_reroute])
    print(f"Average wait time was {np.mean(wait_times2)} for pod2 N = {N}")"""

# print(f"Average of {3600/dt} pods per hour")

congestion = [np.diff(np.sort(G.nodes[node]["reservations"])) for node in G.nodes]
congestion_colormap = []
for node in congestion:
    if node.size == 0:
        congestion_colormap.append(np.nan)
    else:
        congestion_colormap.append(np.mean(node))

congestion_colormap = np.array(congestion_colormap)

################### Here be graph functions

color_map = []

def compute_time_moving_average(pods):
    plt.title(
        "Time to calculate route and wait time per pod against number of pods on the network"
    )
    compute_times = np.array([pod.compute_time for pod in pods])
    window_width = 50
    cumsum = np.cumsum(np.insert(compute_times, 0, 0))
    ma = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    plt.plot(ma)


def wait_time_moving_average(pods):
    plt.title("Wait time per pod against number of pods on the network")
    wait_times = np.array([pod.wait_time for pod in pods])
    window_width = 50
    cumsum = np.cumsum(np.insert(wait_times, 0, 0))
    ma = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    plt.plot(ma)

def wait_time_by_rate(pods):
    plt.title("Wait time per pod against pods per hour")
    plt.plot(wait_plot_frequency, wait_plot1, label="Delay Only")
    #plt.plot(wait_plot_frequency, wait_plot2, label="Delay and Reroute")


# Plot the streets
"""cmap = plt.colormaps["turbo"]

norm = plt.Normalize(
    vmin=np.nanmin(congestion_colormap), vmax=np.nanmax(congestion_colormap)
)

nc = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(congestion_colormap)

fig, ax = ox.plot_graph(
    G, node_color=nc, node_size=100, bgcolor="white", show=False, close=False
)

cb = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal"
)"""

"""for node in master_list: # annotates the node IDs
    text = node
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]
    ax.annotate(text, (x, y), c='y')"""

######################  THE GRAPH SUMMONING ZONE

#fig, ax = ox.plot_graph_routes(G, routes_wait, bgcolor = "white", route_linewidth=6, node_size=0)
#fig, ax = ox.plot_graph_route(G, routes_wait[0], bgcolor = "white", route_linewidth=6, node_size=0)

# compute_time_moving_average(pods)
# wait_time_moving_average(pods)
# wait_time_by_rate(pods)

plt.show()