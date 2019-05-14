#
# This file provides the source code of the Single-Pole balancing experiment using on NEAT-Python library
#

# The Python standard library import
import os
import shutil
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
# The cart-pole simulator
import cart_pole as cart

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    XOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
        n_generations: The numer of generations of evolution
                process to run
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = cart.eval_fitness(net)

def run_experiment(config_file, n_generations=100):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/spb-neat-checkpoint-'))

    # Run for up to N generations.
    best_genome = p.run(eval_genomes, n=n_generations)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Check if the best genome is a winning Sinle-Pole balancing controller 
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    best_genome_fitness = cart.eval_fitness(net)
    if best_genome_fitness >= config.fitness_threshold:
        print("\n\nSUCCESS: The Sinle-Pole balancing controller found!!!")
    else:
        print("\n\nFAILURE: Failed to find Sinle-Pole balancing controller!!!")

    # Visualize the experiment results
    node_names = {-1:'X1', -2: 'X2', -3: 'X3', -4: 'X4', 0:'action'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'single_pole_config.ini')

    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Run the experiment
    run_experiment(config_path)