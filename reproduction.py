import math
import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig

MAX_AGE = 20


# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 1)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            remaining_species.append(stag_s)

        new_population = {}
        for s in remaining_species:

            old_members = []
            all_old_members = []
            for i, v in s.members.items():
                if v.fitness >= 2:
                    old_members.append((i, v))
                elif v.fitness >= 1:
                    all_old_members.append((i, v))

            species.species[s.key] = s
            for i, m in all_old_members:
                if generation - m.createdGeneration < MAX_AGE:
                    new_population[i] = m

            s.members = {}

            for x in old_members:
                for y in range(math.floor(x[1].fitness - 1)):

                    parent1_id, parent1 = x
                    parent2_id, parent2 = random.choice(old_members)

                    gid = next(self.genome_indexer)
                    child = config.genome_type(gid)
                    child.configure_crossover(parent1, parent2, generation, config.genome_config)
                    child.mutate(config.genome_config)

                    new_population[gid] = child
                    self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
