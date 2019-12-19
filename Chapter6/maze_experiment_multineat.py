#
# The script to run maze navigation experiment with Novelty Search optimization
# using the MultiNEAT library
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

# The MultiNEAT specific
import MultiNEAT as NEAT
from MultiNEAT.viz import Draw

# The helper used to visualize experiment results
import visualize
import utils

# The maze environment
import maze_environment as maze
import agent
import novelty_archive as archive

# The number of maze solving simulator steps
SOLVER_TIME_STEPS = 400

class ANN:
    """
    The wrapper of MultiNEAT NeuralNetwork class
    """
    def __init__(self, multi_neat_nn):
        """
        Creates new instance of the wrapper for a given NeuralNetwork
        """
        self.nn = multi_neat_nn

    def activate(self, inputs):
        """
        Function to activate associated NeuralNetwork with given inputs
        Argumnets:
            inputs: the array with network inputs.
        Returns:
            The control signal outputs.
        """
        # append bias
        inputs.append(1.0)
        # activate and get outputs
        self.nn.Input(inputs)
        self.nn.Activate()
        return self.nn.Output()

class Genome:
    def __init__(self, gen):
        self.genome = gen
        self.key = gen.GetID()

class MazeSimulationTrial:
    """
    The class to hold maze simulator execution parameters and results.
    """
    def __init__(self, maze_env, population, archive):
        """
        Creates new instance and initialize fileds.
        Arguments:
            maze_env:   The maze environment as loaded from configuration file.
            population: The population for this trial run
            archive:    The archive to hold NoveltyItems
        """
        # The initial maze simulation environment
        self.orig_maze_environment = maze_env
        # The record store for evaluated maze solver agents
        self.record_store = agent.AgentRecordStore()
        # The NEAT population object
        self.population = population
        # The NoveltyItem archive
        self.archive = archive

def eval_individual(genome_id, genome, genomes, n_items_map, generation):
    """
    Evaluates the individual represented by genome.
    Arguments:
        genome_id:      The ID of genome.
        genome:         The genome to evaluate.
        genomes:        The genomes population for current generation.
        n_items_map:    The map to hold novelty items for current generation.
        generation:     The current generation.
    Return:
        The True if successful solver found.
    """
    # create NoveltyItem for genome and store it into map
    n_item = archive.NoveltyItem(generation=generation, genomeId=genome_id)
    n_items_map[genome_id] = n_item
    # run the simulation
    maze_env = copy.deepcopy(trial_sim.orig_maze_environment)
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)
    control_net = ANN(multi_net)
    goal_fitness = maze.maze_simulation_evaluate(
                                        env=maze_env, 
                                        net=control_net, 
                                        time_steps=SOLVER_TIME_STEPS,
                                        n_item=n_item)

    # Store simulation results into the agent record
    record = agent.AgentRecord(generation=generation, agent_id=genome_id)
    record.fitness = goal_fitness
    record.x = maze_env.agent.location.x
    record.y = maze_env.agent.location.y
    record.hit_exit = maze_env.exit_found
    #record.species_id = trial_sim.population.species.get_species_id(genome_id)
    #record.species_age = record.generation - trial_sim.population.species.get_species(genome_id).created
    # add record to the store
    trial_sim.record_store.add_record(record)

    # Evaluate the novelty of a genome and add the novelty item to the archive of Novelty items if appropriate
    if not maze_env.exit_found:
        # evaluate genome novelty and add it to the archive if appropriate
        record.novelty = trial_sim.archive.evaluate_individual_novelty(genome=Genome(genome), 
                                                                        genomes=genomes, n_items_map=n_items_map)

    # update fittest organisms list
    trial_sim.archive.update_fittest_with_genome(genome=Genome(genome), n_items_map=n_items_map)

    return (maze_env.exit_found, goal_fitness)

def eval_genomes(genomes, generation):
    n_items_map = {} # The map to hold the novelty items for current generation
    solver_genome = None
    best_genome = None
    max_fitness = 0
    for _, genome in genomes:
        found, goal_fitness = eval_individual(genome_id=genome.GetID(), 
                                                genome=genome, 
                                                genomes=genomes, 
                                                n_items_map=n_items_map, 
                                                generation=generation)
        if found:
            solver_genome = genome
            max_fitness = goal_fitness
        elif goal_fitness > max_fitness:
            max_fitness = goal_fitness
            best_genome = genome

    # now adjust the archive settings and evaluate population
    trial_sim.archive.end_of_generation()
    for _, genome in genomes:
        # set fitness value as a logarithm of a novelty score of a genome in the population
        fitness = trial_sim.archive.evaluate_individual_novelty(genome=Genome(genome),
                                                                genomes=genomes,
                                                                n_items_map=n_items_map,
                                                                only_fitness=True)

        # assign the adjusted fitness score to the genome
        genome.SetFitness(fitness)

    if solver_genome is not None:
        return (solver_genome, True, max_fitness)
    else:
        return (best_genome, False, max_fitness)

def run_experiment(params, maze_env, novelty_archive, trial_out_dir, args=None, n_generations=100, 
                    save_results=False, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        params:             The NEAT parameters
        maze_env:           The maze environment to use in simulation.
        novelty_archive:    The archive to work with NoveltyItems.
        trial_out_dir:      The directory to store outputs for this trial
        n_generations:      The number of generations to execute.
        save_results:       The flag to control if intermdiate results will be saved.
        silent:             If True than no intermediary outputs will be
                            presented until solution is found.
        args:               The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """
    # set random seed
    seed = int(time.time())#1562938287#42#1563358622#1559231616#
    random.seed(seed)

    # Create Population
    genome = NEAT.Genome(0, 11, 0, 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, seed)  

    # Create the trial simulation
    global trial_sim
    trial_sim = MazeSimulationTrial(maze_env=maze_env, population=pop, archive=novelty_archive)

    # Run for up to N generations.
    start_time = time.time()
    best_genome_ser = None
    best_ever_goal_fitness = 0
    best_id = -1
    solution_found = False

    for generation in range(n_generations):
        gen_time = time.time()
        # get list of current genomes
        genomes = NEAT.GetGenomeList(pop)
        genomes_tuples = []
        for genome in genomes:
            genomes_tuples.append((genome.GetID(), genome))

        # evaluate genomes
        genome, solution_found, fitness = eval_genomes(genomes_tuples, generation)

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
        print("\n****** Generation: %d ******\n" % generation)
        print("Best objective fitness: %f, genome ID: %d" % (fitness, best_id))
        print("Species count: %d" % len(pop.Species))
        print("Generation elapsed time: %.3f sec" % (gen_elapsed_time))
        print("Best objective fitness ever: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
        print("Best novelty score: %f, genome ID: %d\n" % (pop.GetBestFitnessEver(), pop.GetBestGenome().GetID()))

    elapsed_time = time.time() - start_time

    best_genome = pickle.loads(best_genome_ser)

    # write best genome to the file
    best_genome_file = os.path.join(trial_out_dir, "best_genome.pickle")
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_genome, genome_file)

    # write the record store data
    rs_file = os.path.join(trial_out_dir, "data.pickle")
    trial_sim.record_store.dump(rs_file)

    print("Record store file: %s" % rs_file)
    print("Random seed:", seed)
    print("Trial elapsed time: %.3f sec" % (elapsed_time))
    print("Best objective fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_genome.GetID()))
    print("Best novelty score: %f, genome ID: %d\n" % (pop.GetBestFitnessEver(), pop.GetBestGenome().GetID()))

    # Visualize the experiment results
    show_results = not silent
    if save_results or show_results:
        if args is None:
            visualize.draw_maze_records(maze_env, trial_sim.record_store.records, view=show_results)
        else:
            visualize.draw_maze_records(maze_env, trial_sim.record_store.records, 
                                        view=show_results, 
                                        width=args.width,
                                        height=args.height,
                                        filename=os.path.join(trial_out_dir, 'maze_records.svg'))
        # store NoveltyItems archive data
        trial_sim.archive.write_fittest_to_file(path=os.path.join(trial_out_dir, 'ns_items_fittest.txt'))
        trial_sim.archive.write_to_file(path=os.path.join(trial_out_dir, 'ns_items_all.txt'))

        # create the best genome simulation path and render
        maze_env = copy.deepcopy(trial_sim.orig_maze_environment)
        multi_net = NEAT.NeuralNetwork()
        best_genome.BuildPhenotype(multi_net)
        control_net = ANN(multi_net)
        path_points = []
        evaluate_fitness = maze.maze_simulation_evaluate(
                                    env=maze_env, 
                                    net=control_net, 
                                    time_steps=SOLVER_TIME_STEPS,
                                    path_points=path_points)
        print("Evaluated fitness: %f, of best agent ID: %d" % (evaluate_fitness, best_genome.GetID()))
        visualize.draw_agent_path(trial_sim.orig_maze_environment, path_points, Genome(best_genome),
                                    view=show_results, 
                                    width=args.width,
                                    height=args.height,
                                    filename=os.path.join(trial_out_dir, 'best_solver_path.svg'))

    return solution_found

def create_params():
    params = NEAT.Parameters()
    params.PopulationSize = 500 # 250
    params.DynamicCompatibility = True
    params.AllowClones = False
    params.AllowLoops = True
    params.CompatTreshold = 6.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 200
    params.MinSpecies = 3
    params.MaxSpecies = 20
    params.RouletteWheelSelection = True

    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.3

    params.LinkTries = 40
    params.SpeciesDropoffAge = 200
    params.DisjointCoeff = 1.0
    params.ExcessCoeff = 1.0

    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 0.8
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    #params.MaxWeight = 8

    params.MutateAddNeuronProb = 0.1
    params.MutateAddLinkProb = 0.5
    params.MutateRemLinkProb = 0.1

    params.Elitism = 0.1

    params.CrossoverRate = 0.2
    params.MultipointCrossoverRate = 0.6
    params.InterspeciesCrossoverRate = 0.01

    params.MutateNeuronTraitsProb = 0.1
    params.MutateLinkTraitsProb = 0.1

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment runner (Novelty Search).")
    parser.add_argument('-m', '--maze', default='medium', 
                        help='The maze configuration to use.')
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=1, help='The number of trials to run')
    parser.add_argument('-n', '--ns_threshold', type=float, default=6.0,
                        help="The novelty threshold value for the archive of NoveltyItems.")
    parser.add_argument('-r', '--location_sample_rate', type=int, default=4000,
                        help="The sample rate of agent position points saving during simulation steps.")
    parser.add_argument('--width', type=int, default=400, help='The width of the records subplot')
    parser.add_argument('--height', type=int, default=400, help='The height of the records subplot')
    args = parser.parse_args()

    if not (args.maze == 'medium' or args.maze == 'hard'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    # The current working directory
    local_dir = os.path.dirname(__file__)
    # The directory to store outputs
    out_dir = os.path.join(local_dir, 'out')
    out_dir = os.path.join(out_dir, 'maze_ns_multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the experiment
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    maze_env = maze.read_environment(maze_env_config)
    maze_env.location_sample_rate = args.location_sample_rate

    # Run the maze experiment trials
    print("Starting the %s maze experiment (Novelty Search) with MultiNEAT, for %d trials" % (args.maze, args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        # Create novelty archive
        novelty_archive = archive.NoveltyArchive(threshold=args.ns_threshold,
                                                 metric=maze.maze_novelty_metric)
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        solution_found = run_experiment( params=create_params(), 
                                        maze_env=maze_env, 
                                        novelty_archive=novelty_archive,
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        args=args,
                                        save_results=True,
                                        silent=True)
    print("\n------ Trial %d complete, solution found: %s ------\n" % (t, solution_found))