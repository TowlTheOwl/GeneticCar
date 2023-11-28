import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    x =np.clip(x, -500, 500 )
    return 1/(1 + np.exp(-x))

def graph_data(gen_dist, dist, gen_time, time):
    fig = plt.figure()
    dist_graph = fig.add_subplot(121)
    time_graph = fig.add_subplot(122)

    dist_graph.plot(gen_dist, dist, color='r')
    dist_graph.set_title("Distance")

    time_graph.plot(gen_time, time, color='b')
    time_graph.set_title("Time")

    fig.subplots_adjust(hspace=.5, wspace=.5)

    plt.show()
