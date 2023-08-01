from abc import abstractmethod
from cmath import sqrt

import pygame
import random
import os
import time
import neat
import visualize
import pickle
import numpy

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
gen = 0



class Agent(pygame.sprite.Sprite):
    rotation = 90
    def __init__(self, x, y):
        self.position = numpy.array([x, y])

    def move(self, input):
        self.position = numpy.add(self.position, (numpy.sin(self.rotation) * (input/2+0.25), numpy.cos(self.rotation) * (input/2+0.25)))\
            .clip((10, 10), (SCREEN_WIDTH-10, SCREEN_HEIGHT-10))
    def turn(self, input):
        self.rotation += input / 180
    def draw(self):
        pygame.draw.circle(screen, (255, 255, 255), (self.position[0], self.position[1]), 10)
        pygame.draw.circle(screen, (180, 0, 0), (self.position[0] + numpy.sin(self.rotation + 0.6)*8, self.position[1] + numpy.cos(self.rotation + 0.6)*8), 4)
        pygame.draw.circle(screen, (180, 0, 0), (self.position[0] + numpy.sin(self.rotation - 0.6)*8, self.position[1] + numpy.cos(self.rotation - 0.6)*8), 4)

    def raycast(self, foods, agents):

        r = 8
        ray = numpy.ones(8)
        type = numpy.zeros(8)

        angles = numpy.arange(1, 9) - 4.5
        d = numpy.column_stack((numpy.sin(self.rotation + angles * 0.08), numpy.cos(self.rotation + angles * 0.08))) * 200
        a = numpy.sum(d ** 2, axis=1)
        f = numpy.array([self.position - food.position for food in foods])

        valid_food_indices = numpy.where(numpy.linalg.norm(f, axis=1) < 1000)[0]
        c_values = numpy.sum(f[valid_food_indices] ** 2, axis=1) - r ** 2
        b_values = 2 * numpy.dot(f[valid_food_indices], d.T)
        discriminant_values = b_values ** 2 - 4 * a * c_values[:, numpy.newaxis]
        valid_ray_indices = numpy.where(discriminant_values >= 0)

        t1_values = (-b_values[valid_ray_indices] - numpy.sqrt(discriminant_values[valid_ray_indices])) / (2 * a[valid_ray_indices[1]])

        for val, t1_value in zip(valid_ray_indices[1], t1_values):
            if 0 <= t1_value <= ray[val]:
                ray[val] = t1_value
                pygame.draw.line(screen, (255, 0, 0), self.position, self.position + d[val] * t1_value, 1)



        return ray

class Food(pygame.sprite.Sprite):
    def __init__(self, x, y):
        self.position = numpy.array([x, y])
        super(Food, self).__init__()

    def draw(self):
        pygame.draw.circle(screen, (0, 180, 0), (self.position[0], self.position[1]), 10)


def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file)
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint(os.path.join(local_dir, "neat-checkpoint-24"))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    winner = p.run(eval_genomes)
    print('\nBest genome:\n{!s}'.format(winner))




def eval_genomes(genomes, config):
    global gen
    gen += 1
    nets = []
    agents = []
    foods = []
    ge = []


    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        match genome_id % 4:
            case 0:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), 0))
            case 1:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT))
            case 2:
                agents.append(Agent(0, random.randint(0, SCREEN_HEIGHT)))
            case 3:
                agents.append(Agent(SCREEN_WIDTH, random.randint(0, SCREEN_HEIGHT)))

        ge.append(genome)
    for x in range(20):
        foods.append(Food(random.randint(0, SCREEN_WIDTH),random.randint(0, SCREEN_HEIGHT)))


    clock = pygame.time.Clock()
    i = 1000

    run = True
    while run and i > 0:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        for food in foods:
            food.draw()

        for x, agent in enumerate(agents):
            agent.raycast(foods, agents)
            output = nets[agents.index(agent)].activate((agent.position[0], agent.position[1], agent.rotation))
            agent.move(output[0])
            agent.turn(output[1])
            agent.draw()
        i-=1
        clock.tick(400)
        pygame.display.flip()
        print(clock.get_fps())



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)