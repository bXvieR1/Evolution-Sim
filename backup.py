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
        super(Agent, self).__init__()

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
        type = numpy.zeros((8))

        d = numpy.array([
            [numpy.sin(self.rotation + (x - 4.5) * 0.08) * 200,
             numpy.cos(self.rotation + (x - 4.5) * 0.08) * 200] for x in range(1, 9)
        ])
        #a = numpy.sum(d**2, axis=1)
        a = numpy.array([numpy.dot(x, x) for x in d])
        for food in foods:
            f = self.position - food.position
            if numpy.linalg.norm(f) < 100:
                c = numpy.dot(f, f) - r ** 2
                b = 2 * numpy.dot(f, d.T)
                discriminant = b ** 2 - 4 * a * c
                mask = discriminant >= 0
                discriminant = numpy.sqrt(numpy.maximum(discriminant, 0))  # Ensure non-negative square root
                t1 = (-b - discriminant) / (2 * a)
                mask &= (0 <= t1) & (t1 <= ray)
                type[mask] = 1
                ray[mask] = numpy.minimum(ray[mask], t1[mask])
                for x, hit in enumerate(ray):
                    pygame.draw.line(screen, (255, 255, 255), self.position, numpy.add(self.position, d[x] * ray[x]),1)


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