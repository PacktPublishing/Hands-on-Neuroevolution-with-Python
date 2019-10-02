#
# The script to maintain the modular retina test environment.
#
from enum import Enum
import numpy as np

class Side(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class VisualObject:
    """
    The class to encode the visual object representation
    """
    def __init__(self, configuration, side, size=2):
        """
        Creates new instance with provided configuration and object size
        Arguments:
            configuration:  The configuration of the visual object in form of the text:
                            o o
                            o .
            side:           The side of the retina this object must occupy
            size:           The size of the visual object
        """
        self.size = size
        self.side = side
        self.configuration = configuration
        self.data = np.zeros((size, size))
        
        # Parse configuration
        lines = self.configuration.splitlines()
        for r, line in enumerate(lines):
            chars = line.split(" ")
            for c, ch in enumerate(chars):
                if ch == 'o':
                    # pixel is ON
                    self.data[r, c] = 1.0#3.0
                else:
                    # pixel is OFF
                    self.data[r, c] = 0.0#-3.0

    def get_data(self):
        return self.data.flatten().tolist()
    
    def __str__(self):
        """
        Returns the nicely formatted string representation of this object.
        """
        return "%s\n%s" % (self.side.name, self.configuration)

class RetinaEnvironment:
    """
    Represents the modular retina environment holding test data set and providing
    methods to evaluate detector ANN against it.
    """
    def __init__(self):
        self.visual_objects = []
        # populate data set
        self.create_data_set()
        
    def evaluate_net(self, net, depth = 3, max_fitness = 1000.0, debug=False):
        """
        The function to evaluate performance of the provided network
        against the dataset
        Returns:
            the fitness score and error
        """
        error_sum = 0.0
        count = 0.0
        detection_error_count = 0.0
        # Evaluate the detector ANN against 256 combintaions of the left and the right visual objects
        # at correct and incorrect sides of retina
        for left in self.visual_objects:
            for right in self.visual_objects:
                error, _ = self._evaluate(net, left, right, depth, debug=debug)
                error_sum += error
                count += 1.0
                if error > 0:
                    detection_error_count += 1.0

        
        # calculate the fitness score
        fitness = max_fitness / (1.0 + error_sum)
        avg_error = error_sum / count

        if debug:
            print("Average error: %f, errors sum: %f, false detections: %s" % (avg_error, error_sum, detection_error_count))

        return fitness, avg_error, count, detection_error_count

    def _evaluate(self, net, left, right, depth, debug=False):
        """
        The function to evaluate ANN against specific visual objects at lEFT and RIGHT side
        """
        net.Flush()
        # prepare input
        inputs = left.get_data() + right.get_data()
        inputs.append(0.5) # the bias

        net.Input(inputs)
        # activate
        [net.Activate() for _ in range(depth)]

        # get outputs
        outputs = net.Output()
        outputs[0] = 1.0 if outputs[0] >= 0.5 else 0.0
        outputs[1] = 1.0 if outputs[1] >= 0.5 else 0.0

        # set ground truth
        left_target = 1.0 if left.side == Side.LEFT or left.side == Side.BOTH else 0.0
        right_target = 1.0 if right.side == Side.RIGHT or right.side == Side.BOTH else 0.0
        targets = [left_target, right_target]

        # find error as a distance between outputs and groud truth
        error = (outputs[0] - targets[0]) * (outputs[0] - targets[0]) + \
                (outputs[1] - targets[1]) * (outputs[1] - targets[1])
        flag = "+" if error == 0 else "-"

        if debug:
            print("[%.2f, %.2f] -> [%.2f, %.2f] %s" % (targets[0], targets[1], outputs[0], outputs[1], flag))

        return error, outputs

    def create_data_set(self):
        # set left side objects
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". o\n. o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". o\no o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o o\n. o", side=Side.LEFT))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.BOTH))

        # set right side objects
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o o\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\no o", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.BOTH))

    def __str__(self):
        """
        Returns the nicely formatted string representation of this environment.
        """
        str = "Retina Environment"
        for obj in self.visual_objects:
            str += "\n%s" % obj

        return str
