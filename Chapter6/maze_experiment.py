#
# The script to run maze navigation experiment for both medium and hard
# maze configurations.
#

# The Python standard library import
import os
import shutil
import math
import random
import time
import copy
import argparse

# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
import utils

# The maze environment
import maze_environment as maze
import agent
import novelty_archive as archive

# The number of maze solving simulator steps
SOLVER_TIME_STEPS = 400
# The minimal goal fitness criterion
MCNS = 0.01

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

# The simulation results holder for a one trial.
# It must be initialized before start of each trial.
trial_sim = None

def eval_individual(genome_id, genome, genomes, n_items_map, config):
    """
    Evaluates the individual represented by genome.
    Arguments:
        genome_id:      The ID of genome.
        genome:         The genome to evaluate.
        genomes:        The genomes population for current generation.
        n_items_map:    The map to hold novelty items for current generation.
        config:         The NEAT configuration holder.
    Return:
        The True if successful solver found.
    """
    # create NoveltyItem for genome and store it into map
    n_item = archive.NoveltyItem(generation=trial_sim.population.generation,
                                genomeId=genome_id)
    n_items_map[genome_id] = n_item
    # run the simulation
    maze_env = copy.deepcopy(trial_sim.orig_maze_environment)
    control_net = neat.nn.FeedForwardNetwork.create(genome, config)
    goal_fitness = maze.maze_simulation_evaluate(
                                        env=maze_env, 
                                        net=control_net, 
                                        time_steps=SOLVER_TIME_STEPS,
                                        n_item=n_item,
                                        mcns=MCNS)

    if goal_fitness == -1:
        # The individual doesn't meet the minimal fitness criterion
        print("Individ with ID: %d marked for extiction, MCNS: %f" 
                % (genome_id, MCNS))
        return False

    # Store simulation results into the agent record
    record = agent.AgentRecord(
        generation=trial_sim.population.generation,
        agent_id=genome_id)
    record.fitness = goal_fitness
    record.x = maze_env.agent.location.x
    record.y = maze_env.agent.location.y
    record.hit_exit = maze_env.exit_found
    record.species_id = trial_sim.population.species \
        .get_species_id(genome_id)
    record.species_age = record.generation - \
        trial_sim.population.species.get_species(genome_id).created
    # add record to the store
    trial_sim.record_store.add_record(record)

    # Evaluate the novelty of a genome and add the novelty item to the archive of Novelty items if appropriate
    if not maze_env.exit_found:
        # evaluate genome novelty and add it to the archive if appropriate
        record.novelty = trial_sim.archive \
            .evaluate_individual_novelty(genome=genome, genomes=genomes, 
                                        n_items_map=n_items_map)

    # update fittest organisms list
    trial_sim.archive.update_fittest_with_genome(genome=genome, 
                                        n_items_map=n_items_map)

    return maze_env.exit_found

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    n_items_map = {} # The map to hold the novelty items for current generation
    solver_genome = None
    for genome_id, genome in genomes:
        found = eval_individual(genome_id=genome_id, 
                                genome=genome, 
                                genomes=genomes, 
                                n_items_map=n_items_map, 
                                config=config)
        if found:
            solver_genome = genome

    # now adjust the archive settings and evaluate population
    trial_sim.archive.end_of_generation()
    for genome_id, genome in genomes:
        # set fitness value as a logarithm of a novelty score of a genome in the population
        fitness = trial_sim.archive.evaluate_individual_novelty(genome=genome,
                                                                genomes=genomes,
                                                                n_items_map=n_items_map,
                                                                only_fitness=True)
        # To avoid negative genome fitness scores we just set to zero all obtained
        # fitness scores that is less than 1 (note we use the natural logarithm)
        if fitness > 1:
            fitness = math.log(fitness)
        else:
            fitness = 0
        # assign the adjusted fitness score to the genome
        genome.fitness = fitness

    # if successful maze solver was found then adjust its fitness 
    # to signal the finish evolution
    if solver_genome is not None:
        solver_genome.fitness = math.log(800000) # ~=13.59


def run_experiment(config_file, maze_env, novelty_archive, trial_out_dir, args=None, n_generations=100, 
                    save_results=False, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file:        The path to the file with experiment configuration
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
    seed = int(time.time())
    random.seed(seed)
    print("Selected random seed:", seed)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)    

    # Create the trial simulation
    global trial_sim
    trial_sim = MazeSimulationTrial(maze_env=maze_env, 
                                    population=p,
                                    archive=novelty_archive)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to N generations.
    start_time = time.time()
    best_genome = p.run(eval_genomes, n=n_generations)

    elapsed_time = time.time() - start_time

    # Display the best genome among generations.
    print('\nBest genome:\n%s' % (best_genome))

    solution_found = \
        (best_genome.fitness >= config.fitness_threshold)
    if solution_found:
        print("SUCCESS: The stable maze solver controller was found!!!")
    else:
        print("FAILURE: Failed to find the stable maze solver controller!!!")

    # write the record store data
    rs_file = os.path.join(trial_out_dir, "data.pickle")
    trial_sim.record_store.dump(rs_file)

    print("Record store file: %s" % rs_file)
    print("Random seed:", seed)
    print("Trial elapsed time: %.3f sec" % (elapsed_time))

    # Visualize the experiment results
    show_results = solution_found or not silent
    if save_results or show_results:
        node_names =   {-1:'RF_R', -2:'RF_FR', -3:'RF_F', -4:'RF_FL', -5:'RF_L', -6: 'RF_B', 
                        -7:'RAD_F', -8:'RAD_L', -9:'RAD_B', -10:'RAD_R', 
                        0:'ANG_VEL', 1:'VEL'}
        visualize.draw_net(config, best_genome, view=show_results, node_names=node_names, directory=trial_out_dir, fmt='svg')
        if args is None:
            visualize.draw_maze_records(maze_env, trial_sim.record_store.records, view=show_results)
        else:
            visualize.draw_maze_records(maze_env, trial_sim.record_store.records, 
                                        view=show_results, width=args.width, height=args.height,
                                        filename=os.path.join(trial_out_dir, 'maze_records.svg'))
        visualize.plot_stats(stats, ylog=False, view=show_results, filename=os.path.join(trial_out_dir, 'avg_fitness.svg'))
        visualize.plot_species(stats, view=show_results, filename=os.path.join(trial_out_dir, 'speciation.svg'))

        # store NoveltyItems archive data
        trial_sim.archive.write_fittest_to_file(path=os.path.join(trial_out_dir, 'ns_items_fittest.txt'))
        trial_sim.archive.write_to_file(path=os.path.join(trial_out_dir, 'ns_items_all.txt'))

        # create the best genome simulation path and render
        maze_env = copy.deepcopy(trial_sim.orig_maze_environment)
        control_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        path_points = []
        evaluate_fitness = maze.maze_simulation_evaluate(
                                    env=maze_env, 
                                    net=control_net, 
                                    time_steps=SOLVER_TIME_STEPS,
                                    path_points=path_points)
        print("Evaluated fitness of best agent: %f" % evaluate_fitness)
        visualize.draw_agent_path(trial_sim.orig_maze_environment, path_points, best_genome,
                                    view=show_results, 
                                    width=args.width,
                                    height=args.height,
                                    filename=os.path.join(trial_out_dir, 'best_solver_path.svg'))

    return solution_found

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
    out_dir = os.path.join(out_dir, 'maze_ns')

    # Determine path to configuration file.
    config_path = os.path.join(local_dir, 'maze_config.ini')

    # Clean results of previous run if any or init the ouput directory
    out_dir = os.path.join(out_dir, args.maze)
    utils.clear_output(out_dir)

    # Read the maze environment configuration
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    maze_env = maze.read_environment(maze_env_config)
    maze_env.location_sample_rate = args.location_sample_rate

    # Run the maze experiment trials
    print("Starting the %s maze experiment (Novelty Search), for %d trials" % (args.maze, args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        # Create novelty archive
        novelty_archive = archive.NoveltyArchive(threshold=args.ns_threshold,
                                                 metric=maze.maze_novelty_metric)
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        solution_found = run_experiment( config_file=config_path, 
                                        maze_env=maze_env, 
                                        novelty_archive=novelty_archive,
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        args=args,
                                        save_results=True,
                                        silent=True)
        print("\n------ Trial %d complete, solution found: %s ------\n" % (t, solution_found))