#
# The script providing implementation of structures and functions used in 
# the Novelty Search method.
#
from functools import total_ordering

# how many nearest neighbors to consider for calculating novelty score?
KNN = 15
# the maximal novelty archive size
MAXNoveltyArchiveSize = 1000

@total_ordering
class NoveltyItem:
    """
    The class to encapsulate information about particular item that
    holds information about novelty score associated with specific
    genome along with auxiliary information. It is used in combination
    with NoveltyArchive
    """
    def __init__(self, generation=-1, genomeId=-1, novelty=-1):
        """
        Creates new item with specified parameters.
        Arguments:
            generation: The evolution generation when this item was created
            genomeId:   The ID of genome associated with it
            novelty:    The novelty score of genome
        """
        self.generation = generation
        self.genomeId = genomeId
        self.novelty = novelty
        # Indicates whether this item was already added to the archive
        self.in_archive = False
        # The list holding data points associated with this item that will be used
        # to calculate distance between this item and any other item. This distance
        # will be used to estimate the novelty score associated with the item.
        self.data = []

    def __str__(self):
        """
        The function to create string representation
        """
        return "%s: id: %d, at generation: %d, novelty: %f\tdata: %s" % \
            (self.__class__.__name__, self.genomeId, self.generation, self.novelty, self.data)
    
    def _is_valid_operand(self, other):
        return (hasattr(other, "novelty"))

    def __lt__(self, other):
        """
        Compare if this item is less novel than supplied other item.
        """
        if not self._is_valid_operand(other):
            return NotImplemented

        # less novel is less
        return self.novelty < other.novelty

class NoveltyArchive:
    """
    The novelty archive contains all of the novel items we have encountered thus far.
    """
    def __init__(self, metric):
        """
        Creates new instance with specified novelty threshold and function
        defined novelty metric.
        Arguments:
            metric:     The function to calculate the novelty score of specific genome.
        """
        self.novelty_metric = metric

        # list with all novel items found so far
        self.novel_items = []

    def size(self):
        """
        Returns the size of this archive.
        """
        return len(self.novel_items)

    def evaluate_novelty_score(self, item, n_items_list):
        """
        The function to evaluate novelty score of given novelty item among archive items
        and population items.
        Arguments:
            item:           The novelty item to evaluate
            n_items_list:   The list with novelty items for current population
        """
        # collect distances among archived novelty items
        distances = []
        for n in self.novel_items:
            if n.genomeId != item.genomeId:
                distances.append(self.novelty_metric(n, item))
            else:
                print("Novelty Item is already in archive: %d" % n.genomeId)

        # collect distances to the novelty items in the population
        for p_item in n_items_list:
            if p_item.genomeId != item.genomeId:
                distances.append(self.novelty_metric(p_item, item))

        # calculate average KNN
        distances = sorted(distances) 
        item.novelty = sum(distances[:KNN])/KNN

        # store novelty item
        self._add_novelty_item(item)

        return item.novelty

    def write_to_file(self, path):
        """
        The function to write all NoveltyItems stored in this archive.
        Arguments:
            path: The path to the file where to store NoveltyItems
        """
        with open(path, 'w') as file:
            for ni in self.novel_items:
                file.write("%s\n" % ni)

    def _add_novelty_item(self, item):
        """
        The function to add specified NoveltyItem to this archive.
        Arguments:
            item: The NoveltyItem to be added
        """
        # add item
        item.in_archive = True
        if len(self.novel_items) >= MAXNoveltyArchiveSize:
            # check if this item has higher novelty than last item in the archive (minimal novelty)
            if item > self.novel_items[-1]:
                # replace it
                self.novel_items[-1] = item
        else:
            # just add new item
            self.novel_items.append(item)

        # sort items array in descending order by novelty score
        self.novel_items.sort(reverse=True)