#
# The visualization routines
#
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import graphviz

# The MultiNEAT specific
import MultiNEAT as NEAT

def draw_net(nn, view=False, filename=None, directory=None, node_names=None, node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    # neurons
    for index in range(len(nn.neurons)):
        n = nn.neurons[index]
        node_attrs = None, None
        if n.type == NEAT.NeuronType.INPUT:
            node_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(index, 'lightgray')}
        elif n.type == NEAT.NeuronType.BIAS:
            node_attrs = {'style': 'filled', 'shape': 'diamond', 'fillcolor': node_colors.get(index, 'yellow')}
        elif n.type == NEAT.NeuronType.HIDDEN:
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(index, 'white')}
        elif n.type == NEAT.NeuronType.OUTPUT:
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(index, 'lightblue')}

        # add node with name and attributes
        name = node_names.get(index, str(index))
        dot.node(name, _attributes=node_attrs)

    # connections
    for cg in nn.connections:
        a = node_names.get(cg.source_neuron_idx, str(cg.source_neuron_idx))
        b = node_names.get(cg.target_neuron_idx, str(cg.target_neuron_idx))
        style = 'solid'
        color = 'green' if cg.weight > 0 else 'red'
        width = str(0.1 + abs(cg.weight / 5.0))
        dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, directory, view=view)
    return dot

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
