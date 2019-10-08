import copy
import warnings
import random
import argparse
import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np

import geometry
import agent
import maze_environment as maze

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
    ax1.set_ylabel("Avgerage Distance")
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

def draw_agent_path(maze_env, path_points, genome, filename=None, view=False, show_axes=False, width=400, height=400, fig_height=4):
    """
    The function to draw path of the maze solver agent through the maze.
    Arguments:
        maze_env:       The maze environment configuration.
        path_points:    The list of agent positions during simulation.
        genome:         The genome of solver agent.
        filename:       The name of file to store plot.
        view:           The flag to indicate whether to view plot.
        width:          The width of drawing in pixels
        height:         The height of drawing in pixels
        fig_height:      The plot figure height in inches
    """
    # initialize plotting
    fig, ax = plt.subplots()
    fig.set_dpi(100)
    fig_width = fig_height * (float(width)/float(height )) - 0.2
    #print("Plot figure width: %.1f, height: %.1f" % (fig_width, fig_height))
    fig.set_size_inches(fig_width, fig_height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    ax.set_title('Genome ID: %s, Path Length: %d' % (genome.GetID(), len(path_points)))
    # draw path
    for p in path_points:
        circle = plt.Circle((p.x, p.y), 2.0, facecolor='b')
        ax.add_patch(circle)

    # draw maze
    _draw_maze_(maze_env, ax)

    # turn off axis rendering
    if not show_axes:
        ax.axis('off')

    # Invert Y axis to have coordinates origin at the top left
    ax.invert_yaxis()

    # Save figure to file
    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def draw_maze_records(maze_env, records, best_threshold=0.8, filename=None, view=False, show_axes=False, width=400, height=400, fig_height=7):
    """
    The function to draw maze with recorded agents positions.
    Arguments:
        maze_env:       The maze environment configuration.
        records:        The records of solver agents collected during NEAT execution.
        best_threshold: The minimal fitness of maze solving agent's species to be included into the best ones.
        filename:       The name of file to store plot.
        view:           The flag to indicate whether to view plot.
        width:          The width of drawing in pixels
        height:         The height of drawing in pixels
        fig_height:      The plot figure height in inches
    """
    # find the distance threshold for the best species
    dist_threshold = maze_env.agent_distance_to_exit() * (1.0 - best_threshold)
    # generate color palette and find the best species IDS
    max_sid = 0
    for r in records:
        if r.species_id > max_sid:
            max_sid = r.species_id
    colors = [None] * (max_sid + 1)
    sp_idx = [False] * (max_sid + 1)
    best_sp_idx = [0] * (max_sid + 1)
    for r in records:
        if not sp_idx[r.species_id]:
            sp_idx[r.species_id] = True
            rgb = (random.random(), random.random(), random.random())
            colors[r.species_id] = rgb
        if maze_env.exit_point.distance(geometry.Point(r.x, r.y)) <= dist_threshold:
            best_sp_idx[r.species_id] += 1

    # initialize plotting
    fig = plt.figure()
    fig.set_dpi(100)
    fig_width = fig_height * (float(width)/float(2.0 * height )) - 0.2
    #print("Plot figure width: %.1f, height: %.1f" % (fig_width, fig_height))
    fig.set_size_inches(fig_width, fig_height)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)

    # draw species
    n_best_species = 0
    for i, v in enumerate(best_sp_idx):
        if v > 0:
            n_best_species += 1
            _draw_species_(records=records, sid=i, colors=colors, ax=ax1)
        else:
            _draw_species_(records=records, sid=i, colors=colors, ax=ax2)

    ax1.set_title('fitness >= %.1f, species: %d' % (best_threshold, n_best_species))
    ax2.set_title('fitness < %.1f' % best_threshold)

    # draw maze
    _draw_maze_(maze_env, ax1)
    _draw_maze_(maze_env, ax2)

    # turn off axis rendering
    if not show_axes:
        ax1.axis('off')
        ax2.axis('off')
    # Invert Y axis to have coordinates origin at the top left
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    # Save figure to file
    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def _draw_species_(records, sid, colors, ax):
    """
    The function to draw specific species from the records with
    particular color.
    Arguments:
        records:    The records of solver agents collected during NEAT execution.
        sid:        The species ID
        colors:     The colors table by species ID
        ax:         The figure axis instance
    """
    for r in records:
        if r.species_id == sid:
            circle = plt.Circle((r.x, r.y), 2.0, facecolor=colors[r.species_id])
            ax.add_patch(circle)

def _draw_maze_(maze_env, ax):
    """
    The function to draw maze environment
    Arguments:
        maze_env:   The maze environment configuration.
        ax:         The figure axis instance
    """
    # draw maze walls
    for wall in maze_env.walls:
        line = plt.Line2D((wall.a.x, wall.b.x), (wall.a.y, wall.b.y), lw=1.5)
        ax.add_line(line)

    # draw start point
    start_circle = plt.Circle((maze_env.agent.location.x, maze_env.agent.location.y), 
                                radius=2.5, facecolor=(0.6, 1.0, 0.6), edgecolor='w')
    ax.add_patch(start_circle)
    
    # draw exit point
    exit_circle = plt.Circle((maze_env.exit_point.x, maze_env.exit_point.y), 
                                radius=2.5, facecolor=(1.0, 0.2, 0.0), edgecolor='w')
    ax.add_patch(exit_circle)

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment visualizer.")
    parser.add_argument('-m', '--maze', default='hard', help='The maze configuration to use.')
    parser.add_argument('-r', '--records', help='The records file.')
    parser.add_argument('-o', '--output', help='The file to store the plot.')
    parser.add_argument('--width', type=int, default=200, help='The width of the subplot')
    parser.add_argument('--height', type=int, default=200, help='The height of the subplot')
    parser.add_argument('--fig_height', type=float, default=7, help='The height of the plot figure')
    parser.add_argument('--show_axes', type=bool, default=False, help='The flag to indicate whether to show plot axes.')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if not (args.maze == 'medium' or args.maze == 'hard'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    # read maze environment
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    maze_env = maze.read_environment(maze_env_config)

    # read agents records
    rs = agent.AgentRecordStore()
    rs.load(args.records)

    # render visualization
    random.seed(42)
    draw_maze_records(maze_env, 
                      rs.records,
                      width=args.width,
                      height=args.height,
                      fig_height=args.fig_height,
                      view=True,
                      show_axes=args.show_axes,
                      filename=args.output)
