import numpy as np
import networkx as nx
import osmnx as ox
########################
import time
import random

import concurrent.futures

def collision_checker(gap_array, junction_time):
    return np.any(
        (gap_array < (junction_time / 2)) & (gap_array > -(junction_time / 2))
    )


def gapsolver(X, gap_array, junction_time):
    gap_trial_array = gap_array + X
    result = collision_checker(gap_trial_array, junction_time)
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
    def __init__(self, name, orig, dest, graph, t, method):
        self.name = name
        self.orig = orig
        self.dest = dest
        self.graph = graph
        self.t = t
        self.speed = 4
        # temp values for comparison
        if method == "wait_only":
            self.temp_route = nx.shortest_path( G=self.graph, source=self.orig, target=self.dest, weight="length")
        if method == "wait_and_reroute":
            self.routes = [route for route in ox.routing.k_shortest_paths(
                self.graph, self.orig, self.dest, k=4, weight="length"
            )]

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
        start = time.perf_counter()
        self.route, self.timetable, self.wait_time, self.triptime = self.compute_route(self.temp_route)

        for i, node in enumerate(self.route):
            if (
                self.graph.nodes[node]["street_count"] > 2
            ):  # prevents seemless nodes causing slowing
                # pods do not need to reserve nodes that are not junctions
                self.graph.nodes[node]["reservations"] = np.append(
                    self.graph.nodes[node]["reservations"], self.timetable[i]
                )

        self.compute_time = time.perf_counter() - start

    def routeplan_wait_and_reroute(self):
        start = time.perf_counter()

        results = []
        for temp_route in self.routes:
            results.append(self.compute_route(temp_route))

        best_result = min(results, key=lambda result: result[2])

        self.route, self.timetable, self.wait_time, self.triptime = best_result

        for i, node in enumerate(self.route):
            if (
                self.graph.nodes[node]["street_count"] > 2
            ):  # prevents seemless nodes causing slowing
                # pods do not need to reserve nodes that are not junctions
                self.graph.nodes[node]["reservations"] = np.append(
                    self.graph.nodes[node]["reservations"], self.timetable[i]
                )
        self.compute_time = time.perf_counter() - start

    
class fleet:
    def __init__(self, graph, N, master_list, dt, method):
        self.graph = graph
        self.method = method
        self.podlist = []
        self.nodelist = list(self.graph.nodes.keys())
        self.initialize_pods(master_list, N, dt)

    """def initialize_pods(self, graph, master_list, N, dt):
        t = 0
        for pod_number in range(N):
            orig, dest = random.sample(master_list, 2)
            self.podlist.append(nifti(pod_number, orig, dest, graph, t, self.method))
            t += dt"""

    def initialize_pods(self, master_list, N, dt):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit pod initialization tasks to the executor
            futures = [executor.submit(self.initialize_pod, pod_number, master_list, dt) for pod_number in range(N)]
            
            # Collect the initialized pods
            for future in concurrent.futures.as_completed(futures):
                self.podlist.append(future.result())

    def initialize_pod(self, pod_number, master_list, dt):
        orig = random.choice(self.nodelist)
        dest = random.choice(master_list)
        t = pod_number * dt
        return nifti(pod_number, orig, dest, self.graph, t, self.method)
    
    def generate_routes(self):
        self.routes = []
        for pod in self.podlist:
            if self.method == "wait_only":
                pod.routeplan_wait_only()
            if self.method == "wait_and_reroute":
                pod.routeplan_wait_and_reroute()
            self.routes.append(pod.route)
        self.avg_wait = np.mean(np.array([pod.wait_time for pod in self.podlist]))