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
    def __init__(self, generation=-1, genomeId=-1, fitness=-1, novelty=-1):
        """
        Creates new item with specified parameters.
        Arguments:
            generation: The evolution generation when this item was created
            genomeId:   The ID of genome associated with it
            fitness:    The goal-oriented fitness score of genome associated with this item
            novelty:    The novelty score of genome
        """
        self.generation = generation
        self.genomeId = genomeId
        self.fitness = fitness
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
        return "%s: id: %d, at generation: %d, fitness: %f, novelty: %f\tdata: %s" % \
            (self.__class__.__name__, self.genomeId, self.generation, self.fitness, self.novelty, self.data)
    
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

    def evaluate_individual_novelty(self, genome, genomes, n_items_map, generation, only_fitness=False):
        """
        The function to evaluate the novelty score of a single genome within
        population and update its fitness if appropriate (only_fitness=True)
        Arguments:
            genome:         The genome to evaluate
            genomes:        The current population of genomes
            n_items_map:    The map of novelty items for the current population by genome ID
            generation:     The current evolution epoch
            only_fitness:   The flag to indicate if only fitness should be calculated and assigned to genome
                            using the novelty score. Otherwise novelty score will be used to accept
                            genome into novelty items archive.
        Returns:
            The calculated novelty score for individual genome.
        """
        if genome.key not in n_items_map:
            print("WARNING! Found Genome without novelty point associated: %s" +
                "\nNovelty evaluation will be skipped for it. Probably winner found!" % genome.key)
            return
        
        item = n_items_map[genome.key]
        # Check if individual was marked for extinction due to failure to meet minimal fitness criterion
        if item.fitness == -1.0:
            return -1.0

        result = 0.0
        if only_fitness:
            # assign genome fitness according to the average novelty within archive and population
            result = self._novelty_avg_knn(item=item, genomes=genomes, n_items_map=n_items_map)
        else:
            # consider adding a NoveltyItem to the archive based on the distance to a closest neighbors
            result = self._novelty_avg_knn(item=item, n_items_map=n_items_map)
            self._add_novelty_item(item)

        # store found values to the novelty item
        item.novelty = result
        item.generation = generation

        return result

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
        if len(self.novel_items) > MAXNoveltyArchiveSize:
            # check if this item has higher novelty than last item in the archive (minimal novelty)
            if item > self.novel_items[-1]:
                # replace it
                self.novel_items[-1] = item
        else:
            # just add new item
            self.novel_items.append(item)

        # sort items array in descending order by novelty score
        self.novel_items.sort(reverse=True)

    def _map_novelty(self, item):
        """
        The function to map the novelty metric across the archive against provided item
        Arguments:
            item: The NoveltyItem to be used for archive mapping.
        Returns:
            The list with distances (novelty scores) of provided item from items stored in this archive.
        """
        distances = [None] * len(self.novel_items)
        for i, n in enumerate(self.novel_items):
            distances[i] = self.novelty_metric(n, item)

        return distances

    def _map_novelty_in_population(self, item, genomes, n_items_map):
        """
        The function to map the novelty metric across the archive and the current population
        against the provided item.
        Arguments:
            item:        The NoveltyItem to be used for archive mapping.
            genomes:     The list of genomes from current population.
            n_items_map: The map of novelty items for the current population by genome ID.
        Returns:
            The list with distances (novelty scores) of provided item from items stored in this archive
            and from the novelty items associated with genomes in current population.
        """
        # first, map item against the archive
        distances = self._map_novelty(item)

        # second, map item against the population
        for genome_id, _ in genomes:
            if genome_id in n_items_map:
                gen_item = n_items_map[genome_id]
                distance = self.novelty_metric(gen_item, item)
                distances.append(distance)

        return distances

    def _novelty_avg_knn(self, item, n_items_map, genomes=None):
        """
        The function to calculate the novelty score of a given item within the provided population if any
        using a K-nearest neighbor algorithm.
        Argumnets:
            item:           The NoveltyItem to calculate the score
            n_items_map:    The map of novelty items for the current population by genome ID
            genomes:        The list of genomes from population or None. If genomes list provided novelty will be calculated 
                            among population and archive.
        Returns:
            The density within the vicinity of the provided NoveltyItem calculated using the K-nearest neighbor
            algorithm. This density can be used either as a novelty score value or as a fitness value.
        """
        distances = None
        if genomes is not None:
            distances = self._map_novelty_in_population(item=item, genomes=genomes, n_items_map=n_items_map)
        else:
            distances = self._map_novelty(item=item)

        # Nothing to process
        if len(distances) == 0:
            return 0

        # sort by distance (novelty) in ascending order - the minimal first
        distances = sorted(distances)

        # adjust number of neighbors
        neighbors = KNN
        if neighbors > len(distances):
            # the number of mapped distances is less than number of neighbors
            neighbors = len(distances)

        avg_nn = sum(distances[:neighbors])
        density = avg_nn / float(neighbors)
        return density