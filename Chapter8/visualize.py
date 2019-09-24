#
# The visualization routines
#
import warnings

import matplotlib.pyplot as plt

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
    avg_distance = statistics.get_error_mean()
    stdev_distance = statistics.get_error_stdev()

    fig, ax1 = plt.subplots()
    # Plot average distance
    ax1.plot(generation, avg_distance, 'b--', label="average distance")
    ax1.plot(generation, avg_distance - stdev_distance, 'g-.', label="-1 sd")
    ax1.plot(generation, avg_distance + stdev_distance, 'g-.', label="+1 sd")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Avgerage Error")
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
