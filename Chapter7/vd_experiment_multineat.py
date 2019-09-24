#
# The script to run Visual Discrimination experiment using HyperNEAT method
# from the MultiNEAT library
#

# The Python standard library import
import os
import shutil
import math
import random
import time
import copy
import argparse
import pickle

import numpy as np
import cv2

# The MultiNEAT specific
import MultiNEAT as NEAT

# The test environment
import vd_environment as vd_env

# The helper used to visualize experiment results
import utils
from utils import Statistics
import visualize


# The fitness threshold
FITNESS_THRESHOLD = 1.0

def eval_individual(genome, substrate, vd_environment):
    """
    The funtciton to evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        vd_environment: The test visual discrimination environment
    Returns:

    """
    #substrate.PrintInfo()
    # Create ANN from provided CPPN genome and substrate
    net = NEAT.NeuralNetwork()
    genome.BuildHyperNEATPhenotype(net, substrate)

    fitness, dist = vd_environment.evaluate_net(net)
    return fitness, dist
    

def eval_genomes(genomes, substrate, vd_environment, generation):
    """
    The function to evaluate fitness of the entire population against test 
    visual descriminator environment using the provided substrate 
    configuration of the descriminatorANN
    Arguments:
        genomes:        The list of genomes in the population
        substrate:      The substrate configuration of the descriminatorANN
        vd_environment: The test visual descriminator environment
        generation:     The id of current generation
    Returns:
        the tuple (best_genome, max_fitness, distances) with best CPPN genome, 
        the maximal fitness score value and the list of erro distances for each genome
    """
    best_genome = None
    max_fitness = 0
    distances = []
    for genome in genomes:
        fitness, dist = eval_individual(genome, substrate, vd_environment)
        genome.SetFitness(fitness)
        distances.append(dist)

        if fitness > max_fitness:
            max_fitness = fitness
            best_genome = genome
    
    return best_genome, max_fitness, distances


def run_experiment(params, vd_environment, trial_out_dir, num_dimensions, n_generations=100, 
                    save_results=False, silent=False, args=None):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        params:             The NEAT parameters
        vd_environment:     The environment to test visual discrimination
        trial_out_dir:      The directory to store outputs for this trial
        num_dimensions:     The dimensionsionality of visual field
        n_generations:      The number of generations to execute.
        save_results:       The flag to control if intermdiate results will be saved.
        silent:             If True than no intermediary outputs will be
                            presented until solution is found.
        args:               The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """
    # random seed
    seed = int(time.time())

    # Create substrate
    substrate = create_substrate(num_dimensions)

    # Create CPPN genome and population
    g = NEAT.Genome(0,
                    substrate.GetMinCPPNInputs(),
                    0,
                    substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    0,
                    params, 0)

    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    # Run for up to N generations.
    start_time = time.time()
    best_genome_ser = None
    best_ever_goal_fitness = 0
    best_id = -1
    solution_found = False

    stats = Statistics()
    for generation in range(n_generations):
        print("\n****** Generation: %d ******\n" % generation)
        gen_time = time.time()
        # get list of current genomes
        genomes = NEAT.GetGenomeList(pop)

        # evaluate genomes
        genome, fitness, distances = eval_genomes(genomes, vd_environment=vd_environment, 
                                                substrate=substrate, generation=generation)

        stats.post_evaluate(max_fitness=fitness, distances=distances)
        solution_found = fitness >= FITNESS_THRESHOLD
        # store the best genome
        if solution_found or best_ever_goal_fitness < fitness:
            best_genome_ser = pickle.dumps(genome)
            best_ever_goal_fitness = fitness
            best_id = genome.GetID()
        
        if solution_found:
            print('Solution found at generation: %d, best fitness: %f, species count: %d' % (generation, fitness, len(pop.Species)))
            break

        # advance to the next generation
        pop.Epoch()

        # print statistics
        gen_elapsed_time = time.time() - gen_time
        print("Best fitness: %f, genome ID: %d" % (fitness, best_id))
        print("Species count: %d" % len(pop.Species))
        print("Generation elapsed time: %.3f sec" % (gen_elapsed_time))
        print("Best fitness ever: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))

    elapsed_time = time.time() - start_time

    best_genome = pickle.loads(best_genome_ser)

    # write best genome to the file
    best_genome_file = os.path.join(trial_out_dir, "best_genome.pickle")
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_genome, genome_file)

    # Print experiment statistics
    print("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
    print("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    print("Random seed:", seed)

    # Visualize the experiment results
    show_results = not silent
    if save_results or show_results:
        # Draw CPPN network graph
        net = NEAT.NeuralNetwork()
        best_genome.BuildPhenotype(net)
        visualize.draw_net(net, view=show_results, node_names=None, directory=trial_out_dir, fmt='svg')
        print("\nCPPN nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))

         # Visualize activations from the best genome
        net = NEAT.NeuralNetwork()
        best_genome.BuildHyperNEATPhenotype(net, substrate)
        # select random visual field
        index = random.randint(0, len(vd_environment.data_set) - 1)
        print("\nRunning test evaluation against random visual field:", index)
        print("Substrate nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))
        vf = vd_environment.data_set[index]
        # draw activations
        outputs, x, y = vd_environment.evaluate_net_vf(net, vf)
        visualize.draw_activations(outputs, found_object=(x, y), vf=vf,
                                    dimns=num_dimensions, view=show_results, 
                                    filename=os.path.join(trial_out_dir, "best_activations.svg"))

        # Visualize statistics
        visualize.plot_stats(stats, ylog=False, view=show_results, filename=os.path.join(trial_out_dir, 'avg_fitness.svg'))

    return solution_found

def create_substrate(dim):
    """
    The function to create two-sheets substrate configuration with specified
    dimensions of each sheet.
    Arguments:
        dim:    The dimensions accross X, Y axis of the sheet
    """
    # Building sheet configurations of inputs and outputs
    inputs = create_sheet_space(-1, 1, dim, -1)
    outputs = create_sheet_space(-1, 1, dim, 0)

    substrate = NEAT.Substrate( inputs,
                                [], # hidden
                                outputs)

    substrate.m_allow_input_output_links = True

    substrate.m_allow_input_hidden_links = False
    substrate.m_allow_hidden_hidden_links = False
    substrate.m_allow_hidden_output_links = False
    substrate.m_allow_output_hidden_links = False
    substrate.m_allow_output_output_links = False
    substrate.m_allow_looped_hidden_links = False
    substrate.m_allow_looped_output_links = False

    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

    substrate.m_with_distance = True
    substrate.m_max_weight_and_bias = 3.0

    return substrate

def create_sheet_space(start, stop, dim, z):
    """
    The function to create list with coordinates for a specific sheet of the substrate.
    Arguments:
        start:  The start value by particular coordinate axis
        stop:   The stop value by particular coordinate axis
        dim:    The dimensions accross X, Y axis
        z:      The Z coordinatre of this sheet in the substrate.
    """
    lin_sp = np.linspace(start, stop, num=dim)
    space = []
    for x in range(dim):
        for y in range(dim):
            space.append((lin_sp[x], lin_sp[y], z))

    return space

def create_params():
    params = NEAT.Parameters()
    params.PopulationSize = 150

    params.DynamicCompatibility = True
    params.CompatTreshold = 3.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False

    params.MutateRemLinkProb = 0.02
    params.RecurrentProb = 0
    params.OverallMutationRate = 0.15
    params.MutateAddLinkProb = 0.1
    params.MutateAddNeuronProb = 0.03
    params.MutateWeightsProb = 0.90
    params.MaxWeight = 8.0
    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 1.0

    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.5
    params.MinActivationA = 0.05
    params.MaxActivationA = 6.0

    params.MutateNeuronActivationTypeProb = 0.3

    params.ActivationFunction_SignedGauss_Prob = 1.0
    params.ActivationFunction_SignedSigmoid_Prob = 1.0
    params.ActivationFunction_SignedSine_Prob = 1.0
    params.ActivationFunction_Linear_Prob = 1.0
    
    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_TanhCubic_Prob = 0.0 
    params.ActivationFunction_UnsignedStep_Prob = 0.0
    params.ActivationFunction_UnsignedGauss_Prob = 0.0
    params.ActivationFunction_Abs_Prob = 0.0
    params.ActivationFunction_UnsignedSine_Prob = 0.0
    

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    params.AllowLoops = False

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The Visual Discrimination experiment runner (HyperNEAT).")
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=1, help='The number of trials to run')
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)
    # The directory to store outputs
    out_dir = os.path.join(local_dir, 'out')
    out_dir = os.path.join(out_dir, 'vd_multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # create the test environment
    VISUAL_FIELD_SIZE = 11
    env = vd_env.VDEnvironment(small_object_positions=[1, 3, 5, 7, 9],
                                big_object_offset=5,
                                field_size=VISUAL_FIELD_SIZE)

    # Run the maze experiment trials
    print("Starting Visual Discrimination experiment with MultiNEAT, for %d trials" % (args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        soulution_found = run_experiment(params=create_params(),
                                        vd_environment = env,
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        num_dimensions=VISUAL_FIELD_SIZE,
                                        args=args,
                                        save_results=True,
                                        silent=False)
        print("\n------ Trial %d complete, solution found: %s ------\n" % (t, soulution_found))