#
# The script providing implementation of structures and functions used in 
# the Novelty Search method.
#
from functools import total_ordering

@total_ordering
class NoveltyItem:
    """
    The class to encapsulate information about particular item that
    holds information about novelty score associated with specific
    genome along with auxiliary information. It is used in combination
    with NoveltyArchive
    """
    def __init__(self, generation, genomeId, fitness, novelty, age):
        """
        Creates new item with specified parameters.
        Arguments:
            generation: The evolution generation when this item was created
            genomeId:   The ID of genome associated with it
            fitness:    The goal-oriented fitness score of genome
            novelty:    The novelty score of genome
            age:        The age of species holding genome
        """
        self.generation = generation
        self.genomeId = genomeId
        self.fitness = fitness
        self.novelty = novelty
        self.age = age
        # The auxiliary data associated with item
        self.data = []

    def __str__(self):
        """
        The function to create string representation
        """
        return "%s: id: %d, at generation: %d, fitness: %f, novelty: %f, age: %d \tdata: %s" % \
            (self.__class__.__name__, self.genomeId, self.generation, self.fitness, self.novelty, self.age, self.data)
    
    def _is_valid_operand(self, other):
        return (hasattr(other, "fitness") and
                hasattr(other, "novelty"))

    def __lt__(self, other):
        """
        Compare if this item is less than supplied other item by
        goal-oriented fitness value.
        """
        if not self._is_valid_operand(other):
            return NotImplemented

        if self.fitness < other.fitness:
            return True
        elif self.fitness == other.fitness:
            # less novel is less
            return self.novelty < other.novelty
        return False


class ItemsDistance:
    """
    Holds information about distance between the two NoveltyItem objects based
    on the nearest neighbour metric.
    """
    def __init__(self, first_item, second_item, distance):
        """
        Creates new instance for two NoveltyItem objects
        Arguments:
            first_item:     The item from which distance is measured
            second_item:    The item to which distance is measured
            distance:       The distance value
        """
        self.first_item = first_item
        self.second_item = second_item
        self.distance = distance

    def _is_valid_operand(self, other):
        return hasattr(other, "distance")

    def __lt__(self, other):
        """
        Compare if the distance in this object is less that in other.
        """
        if not self._is_valid_operand(other):
            return NotImplemented

        return self.distance < other.distance
