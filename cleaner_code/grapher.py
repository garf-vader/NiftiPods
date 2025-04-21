import matplotlib.pyplot as plt
import numpy as np
import pickle

def compute_time_moving_average(compute_data):
    plt.title(
        "Time to calculate route and wait time per pod against number of pods on the network"
    )
    compute_times = np.array(compute_data)
    window_width = 288
    cumsum = np.cumsum(np.insert(compute_times, 0, 0))
    ma = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    plt.plot(ma)
    plt.savefig(
        "graphs/compute_time_moving_average.pdf", format="pdf", bbox_inches="tight"
    )
    plt.clf()


def wait_time_moving_average(pods): # not including in poster so you can ignore
    plt.title("Wait time per pod against number of pods on the network")
    wait_times = np.array([pod.wait_time for pod in pods.podlist])
    window_width = 50
    cumsum = np.cumsum(np.insert(wait_times, 0, 0))
    ma = (cumsum[window_width:] - cumsum[:-window_width]) / window_width
    plt.plot(ma)
    plt.savefig(
        "graphs/wait_time_moving_average.pdf", format="pdf", bbox_inches="tight"
    )


def wait_time_by_rate(wait_plot_frequency, wait_plot1, wait_plot2):
    plt.title("Wait time per pod against pods per hour")
    plt.plot(wait_plot_frequency, wait_plot1, label="Delay Only")
    plt.plot(wait_plot_frequency, wait_plot2, label="Delay and Reroute")
    plt.legend()
    plt.savefig("graphs/wait_time_by_rate.pdf", format="pdf", bbox_inches="tight")
    plt.clf()


with open("graph_data/compute_time.pickle", "rb") as f:   #unPickling
    compute_data = pickle.load(f)
with open("graph_data/wait_method_comparison.pickle", "rb") as f:   #unPickling
    wait_data = pickle.load(f)

#compute_time_moving_average(compute_data)
wait_time_by_rate(wait_data[0], wait_data[1], wait_data[2])