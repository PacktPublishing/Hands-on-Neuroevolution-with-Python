#
# The visualization routines
#
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def plot_stats(statistics, ylog=False, view=False, filename='avg_distance.svg'):
    """ Plots the population's best fitness and average distances. """

    generation = range(len(statistics.most_fit_scores))
    avg_distance = statistics.get_distance_mean()
    stdev_distance = statistics.get_distance_stdev()

    fig, ax1 = plt.subplots()
    # Plot average distance
    ax1.plot(generation, avg_distance, 'b--', label="average distance")
    ax1.plot(generation, avg_distance - stdev_distance, 'g-.', label="-1 sd")
    ax1.plot(generation, avg_distance + stdev_distance, 'g-.', label="+1 sd")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Distance")
    ax1.grid()
    ax1.legend(loc="best")

    # Plot best fitness
    ax2 = ax1.twinx()
    ax2.plot(generation, statistics.most_fit_scores, 'r-', label="best fitness")
    ax2.set_ylabel("Fitness")

    plt.title("Population's best fitness and average distance")
    fig.tight_layout()
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def draw_activations(activations, found_object, vf, dimns, view=False, filename='activations.svg', fig_width=11):
    """
    Function to plot activations array with specified dimensions.
    """
    print("found", found_object)
    print("target", vf.big_pos)
    # reshape
    data = np.array(activations).reshape((dimns,dimns))

    # render
    grid_kws = {"width_ratios": (.9, .9, .05), "wspace": .2}
    fig, (ax_target, ax_map, cbar_ax) = plt.subplots(nrows=1, ncols=3, gridspec_kw=grid_kws)
    # Draw ANN activations
    sns.heatmap(data, linewidth=0.2, cmap="YlGnBu",
                ax=ax_map, cbar_ax=cbar_ax,
                cbar_kws={"orientation": "vertical"})

    ax_map.set_title("ANN activations map")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")

    # Draw visual field
    sns.heatmap(vf.data, linewidth=0.2, cmap="YlGnBu",
                ax=ax_target, cbar=False)
    ax_target.set_title("Visual field")
    ax_target.set_xlabel("X")
    ax_target.set_ylabel("Y")

    ax_map.set_title("ANN activations map")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")

    # Set figure size
    fig.set_dpi(100)
    fig_height = fig_width / 2.0 - 0.3
    print("Plot figure width: %.1f, height: %.1f" % (fig_width, fig_height))
    fig.set_size_inches(fig_width, fig_height)

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()
