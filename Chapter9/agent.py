#
# This is the definition of a maze navigating agent.
#
import pickle

class Agent:
    """
    This is the maze navigating agent
    """
    def __init__(self, location, heading=0, speed=0, angular_vel=0, radius=8.0, range_finder_range=100.0):
        """
        Creates new Agent with specified parameters.
        Arguments:
            location:               The agent initial position within maze
            heading:                The heading direction in degrees.
            speed:                  The linear velocity of the agent.
            angular_vel:            The angular velocity of the agent.
            radius:                 The agent's body radius.
            range_finder_range:     The maximal detection range for range finder sensors.
        """
        self.heading = heading
        self.speed = speed
        self.angular_vel = angular_vel
        self.radius = radius
        self.range_finder_range = range_finder_range
        self.location = location

        # defining the range finder sensors
        self.range_finder_angles = [-90.0, -45.0, 0.0, 45.0, 90.0, -180.0]

        # defining the radar sensors
        self.radar_angles = [(315.0, 405.0), (45.0, 135.0), (135.0, 225.0), (225.0, 315.0)]

        # the list to hold range finders activations
        self.range_finders = [None] * len(self.range_finder_angles)
        # the list to hold pie-slice radar activations
        self.radar = [None] * len(self.radar_angles)

class AgenRecord:
    """
    The class to hold results of maze navigation simulation for specific
    solver agent. It provides all statistics about the agent at the end
    of navigation run.
    """
    def __init__(self, generation, agent_id):
        """
        Creates new record for specific agent at the specific generation
        of the evolutionary process.
        """
        self.generation = generation
        self.agent_id = agent_id
        # initialize agent's properties
        self.x = -1
        self.y = -1
        self.fitness = -1
        self.novelty = -1
        # The flag to indicate whether this agent was able to find maze exit
        self.hit_exit = False
        # The ID of species this agent belongs to
        self.species_id = -1
        # The age of agent's species at the time of recording
        self.species_age = -1

class AgentRecordStore:
    """
    The class to control agents record store.
    """
    def __init__(self):
        """
        Creates new instance.
        """
        self.records = []

    def add_record(self, record):
        """
        The function to add specified record to this store.
        Arguments:
            record: The record to be added.
        """
        self.records.append(record)

    def load(self, file):
        """
        The function to load records list from the specied file into this class.
        Arguments:
            file: The path to the file to read agents records from.
        """
        with open(file, 'rb') as dump_file:
            self.records = pickle.load(dump_file)

    def dump(self, file):
        """
        The function to dump records list to the specified file from this class.
        Arguments:
            file: The path to the file to hold data dump.
        """
        with open(file, 'wb') as dump_file:
            pickle.dump(self.records, dump_file)
    