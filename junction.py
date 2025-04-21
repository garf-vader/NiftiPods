import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import pandas as pd

G = nx.DiGraph()

G.add_node("A",pos=(0,3))
G.add_node("B",pos=(1,3))
G.add_node("C",pos=(4,3))
G.add_node("D",pos=(5,3))
G.add_node("E",pos=(0,2))
G.add_node("F",pos=(1,2))
G.add_node("G",pos=(4,2))
G.add_node("H",pos=(5,2))
G.add_node("I",pos=(2,1))
G.add_node("J",pos=(3,1))
G.add_node("K",pos=(2,0))
G.add_node("L",pos=(3,0))


buffer = 30

G.add_edge("A", "B", length = buffer, junction = False)
G.add_edge("C", "D", length = buffer, junction = False)
G.add_edge("F", "E", length = buffer, junction = False)
G.add_edge("H", "G", length = buffer, junction = False)
G.add_edge("K", "I", length = buffer, junction = False)
G.add_edge("J", "L", length = buffer, junction = False)

G.add_edge("B", "C", length = 3.3, junction = True, reserved = [pd.Interval(29, 31)], collide = ["I", "C"])
G.add_edge("G", "F", length = 3.3, junction = True, reserved = [])

G.add_edge("I", "F", length = 1.8, junction = True, reserved = [])
G.add_edge("G", "J", length = 1.8, junction = True, reserved = [])

G.add_edge("B", "J", length = 3.4, junction = True, reserved = [])
G.add_edge("I", "C", length = 3.4, junction = True, reserved = [])

# overlap [(B,C),(I,C)], [(),()]

global timestamp

timestamp = 0

reservation = pd.Interval(30, 33.3)

for u, v, a in G.edges(data = True):
    if a["junction"]:
        for res in a["reserved"]:
            print(reservation.overlaps(res))


class Nifti:
    def __init__(self, name, source, target):
        self.name = name
        self.source = source
        self.target = target

    def find_route(self):
        path = nx.shortest_path(G, self.source, self.target, weight = "length")
        time = timestamp
        timings = [timestamp]
        for i, j in zip(path[1:], path):
            length = G.edges[j, i]["length"]
            if G.edges[j, i]["junction"]:
                G.edges[j, i]["reserved"].append((time, time + length))
                time += length
            else: time += length
            timings.append(time)
        self.route = [path, timings]

p1 = Nifti("John", "A", "D")
p1.find_route()

pos=nx.get_node_attributes(G,'pos')

## TO DO
## Avoid reserved intersections / routes
## Link junctions that collide and create an update loop that when one junction is reserved then all the linked ones are updated afterwards

nx.draw(G, pos, with_labels=True)
plt.show()