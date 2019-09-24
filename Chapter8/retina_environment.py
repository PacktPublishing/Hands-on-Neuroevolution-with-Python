#
# The script to maintain the modular retina test environment.
#
from enum import Enum
import numpy as np

class Side(Enum):
    LEFT = 1
    RIGHT = 2

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
                    self.data[r, c] = 3.0
                else:
                    # pixel is OFF
                    self.data[r, c] = -3.0

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
        

    def create_data_set(self):
        # set left side objects
        self.visual_objects.append(VisualObject(". o\no o", side=Side.LEFT))
        self.visual_objects.append(VisualObject("o .\no o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\no o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.LEFT))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.LEFT))

        # set right side objects
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject("o o\n. o", side=Side.RIGHT))
        self.visual_objects.append(VisualObject("o o\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject("o o\n. .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.RIGHT))

    def __str__(self):
        """
        Returns the nicely formatted string representation of this environment.
        """
        str = "Retina Environment"
        for obj in self.visual_objects:
            str += "\n%s" % obj

        return str
