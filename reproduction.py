import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig

MAX_AGE = 5
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
        #species.species = {}
        for s in remaining_species:
            old_members = [(i, v) for i, v in s.members.items() if v.fitness == 1]

            species.species[s.key] = s
            for i, m in old_members:
                if generation - m.createdGeneration < MAX_AGE:
                    new_population[i] = m


            s.members = {}

            # Randomly choose parents and produce the number of offspring allotted to the species.
            for i in range(len(old_members)):

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, generation, config.genome_config)
                child.mutate(config.genome_config)
                # TODO: if config.genome_config.feed_forward, no cycles should exist
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population