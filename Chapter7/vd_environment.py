#
# The script to maintain the visual discriminator environment
#
import math

import numpy as np


class VisualField:
    """
    Represents visual field
    """
    def __init__(self, big_pos, small_pos, field_size):
        self.big_pos = big_pos
        self.small_pos = small_pos
        self.field_size = field_size
        self.data = np.zeros((field_size, field_size))

        # store small object position
        self._set_point(small_pos[0], small_pos[1])

        # store big object points
        offsets = [-1, 0, 1]
        for xo in offsets:
            for yo in offsets:
                self._set_point(big_pos[0] + xo, big_pos[1] + yo)

    def get_data(self):
        return self.data.flatten().tolist()

    def _set_point(self, x, y):
        px, py = x, y
        if px < 0:
            px = self.field_size + px
        elif px >= self.field_size:
            px = px - self.field_size

        if py < 0:
            py = self.field_size + py
        elif py >= self.field_size:
            py = py - self.field_size

        self.data[py, px] = 1 # in Numpy index is: [row, col]

class VDEnvironment:
    """
    Represents test environment to hold data set of visual fields
    """
    def __init__(self, small_object_positions, big_object_offset, field_size):
        self.s_object_pos = small_object_positions
        self.data_set = []
        self.b_object_offset = big_object_offset
        self.field_size = field_size

        self.max_dist = self._distance((0, 0), (field_size - 1, field_size - 1))

        # create test data set
        self._create_data_set()

    def evaluate_net(self, net):
        """
        The function to evaluate performance of the provided network
        against the dataset
        Returns:
            the fitness score and average Euclidean distance between found and target objects
        """
        avg_dist = 0

        # evaluate predicted positions
        for ds in self.data_set:
            # evaluate and get outputs
            _, x, y = self.evaluate_net_vf(net, ds)

            # find the distance to the big object
            dist = self._distance((x, y), ds.big_pos)
            avg_dist = avg_dist + dist

        avg_dist /= float(len(self.data_set))
        
        # normalized detection error
        error = avg_dist / self.max_dist
        # fitness
        fitness = 1.0 - error

        return fitness, avg_dist

    def evaluate_net_vf(self, net, vf):
        """
        The function to evaluate provided ANN against specific VisualField
        """
        depth = 1 # we just have 2 layers

        net.Flush()
        # prepare input
        inputs = vf.get_data()

        net.Input(inputs)
        # activate
        [net.Activate() for _ in range(depth)]

        # get outputs
        outputs = net.Output()
        # find coordinates of big object
        x, y = self._big_object_coordinates(outputs)

        return outputs, x, y

    def _distance(self, source, target):
        """
        Function to find Euclidean distance between source and target points
        """
        dist = (source[0] - target[0]) * (source[0] - target[0]) + (source[1] - target[1]) * (source[1] - target[1])
        return math.sqrt(dist)

    def _big_object_coordinates(self, outputs):
        max_activation = -100.0
        max_index = -1
        for i, out in enumerate(outputs):
            if out > max_activation:
                max_activation = out
                max_index = i

        # estimate the maximal activation's coordinates
        x = max_index % self.field_size
        y = int(max_index / self.field_size)

        return (x, y)


    def _create_visual_field(self, sx, sy, x_off, y_off):
        bx = sx + x_off
        # 5 point to the right
        if bx >= self.field_size:
            bx = bx - self.field_size # wrap
        by = sy + y_off
        if by >= self.field_size:
            by = by - self.field_size # wrap

        # create visual field
        return VisualField(big_pos=(bx, by), small_pos=(sx, sy), field_size=self.field_size)

    def _create_data_set(self):
        for x in self.s_object_pos:
            for y in self.s_object_pos:
                # diagonal
                vf = self._create_visual_field(x, y, self.b_object_offset, self.b_object_offset)
                self.data_set.append(vf)
                # right
                vf = self._create_visual_field(x, y, x_off=self.b_object_offset, y_off=0)
                self.data_set.append(vf)
                # down
                vf = self._create_visual_field(x, y, x_off=0, y_off=self.b_object_offset)
                self.data_set.append(vf)
    