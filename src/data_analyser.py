import numpy as np
import pickle
import os
from collections import defaultdict

fleet_files = os.listdir("./fleets")

# filename = f"{method}_N_{N}_dt_{dt}_{graph_name}_{next(count):02d}"

# Initialize defaultdicts to store dt and count for each N
dt_wait_sum = defaultdict(float)
dt_reroute_sum = defaultdict(float)

N_count_wait = defaultdict(int)
N_count_reroute = defaultdict(int)

for filename in fleet_files:
    with open(f"./fleets/{filename}", "rb") as f:
        pods = pickle.load(f)

    parts = filename.split('-')
    method = str(parts[0])
    N = int(parts[2])
    dt = float(parts[4])

    if method == "wait_only":
        dt_wait_sum[N] += pods.avg_wait
        N_count_wait[N] += 1
    if method == "wait_and_reroute":
        dt_reroute_sum[N] += pods.avg_wait
        N_count_reroute[N] +=1


#total_time = pod.wait_time + pod.triptime
        
"""with open(f"./fleets/wait_only-N-1-dt-2.00-piedmont_clean-00.pickle", "rb") as f:
        bigN = pickle.load(f)
compute_time = [pod.compute_time for pod in bigN.podlist]
with open("graph_data/compute_time.pickle", "wb") as f:   #Pickling
    pickle.dump(compute_time, f)"""
        
wait_avg_wait = {N: dt_wait_sum[N] / N_count_wait[N] for N in dt_wait_sum}
reroute_avg_wait = {N: dt_reroute_sum[N] / N_count_reroute[N] for N in dt_reroute_sum}
sorted_keys = sorted(wait_avg_wait.keys())
N_list = [N for N in sorted_keys]
wait_avg_wait_list = [wait_avg_wait[N] for N in sorted_keys]
reroute_avg_wait_list = [reroute_avg_wait[N] for N in sorted_keys]
wait_data = [N_list, wait_avg_wait_list, reroute_avg_wait_list]

with open("graph_data/wait_method_comparison.pickle", "wb") as f:   #Pickling
    pickle.dump(wait_data, f)