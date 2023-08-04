import math

import numpy
import pygame
import random
import os
import neat
import numpy as np
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

import genome
import reproduction

RAY_COUNT = 16
SCREEN_WIDTH = 1000
DATA_WIDTH = 400
SCREEN_HEIGHT = 600

ENERGY = 5000

RANGE_FACTOR = 100
SIZE_FACTOR = 5
SPEED_FACTOR = 2

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH + DATA_WIDTH, SCREEN_HEIGHT))
calibri = pygame.font.SysFont('Calibri Bold', 20)
gen = 0


class Agent():

    rotation = 90
    energy = ENERGY

    dead = False
    active = True

    def __init__(self, x, y, genome):
        self.position = np.array([x, y])
        self.size = genome.radius
        self.range = genome.range
        self.speed = genome.speed

        self.need = self.size ** 3 * self.speed ** 2 + self.range
        #self.need = self.size * (1 / self.speed) + 1 / self.range ** 2


    def move(self, input):
        self.position = np.add(self.position, (
            np.sin(self.rotation) * input * SPEED_FACTOR, np.cos(self.rotation) * input * SPEED_FACTOR)).clip((10, 10), (
            SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10))

    def turn(self, input):
        self.rotation += 10 * (input / 180) * SPEED_FACTOR

    def draw(self):
        size = SIZE_FACTOR * self.size
        position = self.position + (DATA_WIDTH, 0)
        if self.active:
            if self.energy <= 0:
                pygame.draw.circle(screen, (80, 80, 80), position, size + 2)
                pygame.draw.circle(screen, (100, 100, 100), position, size)
            else:

                body = numpy.clip((int)(self.speed * 150), 50, 200)

                pygame.draw.circle(screen, (0, 0, 0), position, size + 2)
                pygame.draw.circle(screen, (body, np.abs(150 - body) * 2, 255 - body), position, size)

                eye1 = (position[0] + np.sin(self.rotation + 0.6) * size,
                        position[1] + np.cos(self.rotation + 0.6) * size)
                eye2 = (position[0] + np.sin(self.rotation - 0.6) * size,
                        position[1] + np.cos(self.rotation - 0.6) * size)

                pygame.draw.circle(screen, (10, 10, 10), eye1, size / 2 + 1)
                pygame.draw.circle(screen, (10, 10, 10), eye2, size / 2 + 1)

                if self.range > 1:
                    eyes = numpy.clip(255 - (int)(self.range - 1) * 100, 0, 255)
                    pygame.draw.circle(screen, (255, eyes, eyes), eye1, size / 2)
                    pygame.draw.circle(screen, (255, eyes, eyes), eye2, size / 2)
                else:
                    eyes = numpy.clip(255 - (int)(1 - self.range) * 200, 0, 255)

                    pygame.draw.circle(screen, (eyes, 255, eyes), eye1, size / 2)
                    pygame.draw.circle(screen, (eyes, 255, eyes), eye2, size / 2)
        else:
            if self.dead:
                pygame.draw.circle(screen, (180, 0, 0), position, size + 2)
                pygame.draw.circle(screen, (100, 100, 100), position, size)
            else:
                pygame.draw.circle(screen, (0, 180, 0), position, size + 2)
                pygame.draw.circle(screen, (100, 100, 100), position, size)


def raycast(foods, agents):
    output = []
    positions = np.vstack(([food.position for food in foods], [agent.position for agent in agents]))
    angles = np.arange(1, RAY_COUNT + 1) - (RAY_COUNT + 1) / 2
    r = np.concatenate((np.full(len(foods), 5).astype(int), [agent.size for agent in agents]))

    for subject in agents:
        d = np.column_stack(
            (np.sin(subject.rotation + angles * 0.1),
             np.cos(subject.rotation + angles * 0.1))) * subject.range * RANGE_FACTOR
        f = subject.position - positions
        a = np.sum(d ** 2, axis=1)

        valid_food_indices = np.where(np.linalg.norm(f, axis=1) < subject.range * RANGE_FACTOR)[0]

        c_values = np.sum(f[valid_food_indices] ** 2, axis=1) - r[valid_food_indices] ** 2
        b_values = 2 * np.dot(f[valid_food_indices], d.T)
        t_values = valid_food_indices < len(foods)
        discriminant_values = b_values ** 2 - 4 * a * c_values[:, np.newaxis]
        valid_ray_indices = np.where(discriminant_values >= 0)
        t1_values = (-b_values[valid_ray_indices] - np.sqrt(discriminant_values[valid_ray_indices])) / (
                2 * a[valid_ray_indices[1]])
        valid_t1_indices = (t1_values > 0) & (t1_values < 1)

        agentRay = np.zeros(RAY_COUNT)
        foodRay = np.zeros(RAY_COUNT)

        valid_ray_t1_indices = valid_ray_indices[1][valid_t1_indices]
        valid_ray_t1_values = t1_values[valid_t1_indices]

        agentRay[valid_ray_t1_indices] = valid_ray_t1_values * (
            ~t_values[valid_ray_indices[0][valid_t1_indices]])
        foodRay[valid_ray_t1_indices] = valid_ray_t1_values * t_values[
            valid_ray_indices[0][valid_t1_indices]]

        for x, (f, a) in enumerate(zip(foodRay, agentRay)):
            if f == 0:
                pygame.draw.line(screen, (255, 0, 0), subject.position + (DATA_WIDTH, 0), subject.position + d[x] * a + (DATA_WIDTH, 0), 1)
            else:
                pygame.draw.line(screen, (0, 255, 0), subject.position + (DATA_WIDTH, 0), subject.position + d[x] * f + (DATA_WIDTH, 0), 1)

        output.append(np.concatenate([foodRay, agentRay]))
    return np.vstack(output)


class Food:
    def __init__(self, x, y):
        self.position = np.array([x, y])
        # super(Food, self).__init__()

    def draw(self):
        pygame.draw.circle(screen, (0, 180, 0), (self.position + (DATA_WIDTH, 0)), 4)


def run(config_file):
    config = neat.config.Config(
        genome.DefaultGenome,
        reproduction.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file)
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint(os.path.join(local_dir, "neat-checkpoint-1364"))
    #p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))
    winner = p.run(eval_genomes)
    print('\nBest genome:\n{!s}'.format(winner))


stats = numpy.array([[(float)(1)]*100]*7)
def eval_genomes(genomes, config):

    global gen
    global stats

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
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), random.randint(0, 100), genome))
            case 1:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT - random.randint(0, 100), genome))
            case 2:
                agents.append(Agent(random.randint(0, 100), random.randint(0, SCREEN_HEIGHT), genome))
            case 3:
                agents.append(Agent(SCREEN_WIDTH - random.randint(0, 100), random.randint(0, SCREEN_HEIGHT), genome))

        ge.append(genome)



    GRAPH_COUNT = 6
    GRAPH_BORDER = 10
    GRAPH_OUTLINE = 5
    GRAPH_HEIGHT = (SCREEN_HEIGHT) / GRAPH_COUNT - GRAPH_BORDER
    GRAPH_WIDTH = DATA_WIDTH - GRAPH_BORDER * 2

    #region graph1
    limit = numpy.max(stats[:3]) * 1.1
    graph1 = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
    graph1.fill((200, 200, 200))
    for x in np.arange(0.1, limit, 0.1):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        pygame.draw.line(graph1, (175, 175, 175), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph1, (0, 0, 0), (0, y), (2, y), 2)
    for x in np.arange(0.5, limit, 0.5):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        pygame.draw.line(graph1, (150, 150, 150), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph1, (0, 0, 0), (0, y), ( 4, y), 2)
    for x in np.arange(1, limit, 1):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        number = calibri.render(str(x), False, (0, 0, 0))
        pygame.draw.line(graph1, (140, 140, 140), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph1, (0, 0, 0), (0, y), ( 5, y), 3)
        graph1.blit(number, (6, y - 5))

    points = \
    np.dstack([np.arange(0, GRAPH_WIDTH, GRAPH_WIDTH / 100) * 1.01, GRAPH_HEIGHT - (stats[0] / limit * GRAPH_HEIGHT)])[0]
    pygame.draw.lines(graph1, (255, 0, 0), False, points, 1)

    points = \
    np.dstack([np.arange(0, GRAPH_WIDTH, GRAPH_WIDTH / 100) * 1.01, GRAPH_HEIGHT - (stats[1] / limit * GRAPH_HEIGHT)])[0]
    pygame.draw.lines(graph1, (0, 255, 0), False, points, 1)

    points = \
    np.dstack([np.arange(0, GRAPH_WIDTH, GRAPH_WIDTH / 100) * 1.01, GRAPH_HEIGHT - (stats[2] / limit * GRAPH_HEIGHT)])[0]
    pygame.draw.lines(graph1, (0, 0, 255), False, points, 1)
    # endregion
    #region graph2
    limit = numpy.max(stats[3]) * 1.1
    graph2 = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
    graph2.fill((200, 200, 200))

    for x in np.arange(5, limit, 5):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        pygame.draw.line(graph2, (175, 175, 175), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph2, (0, 0, 0), (0, y), (2, y), 2)
    for x in np.arange(25, limit, 25):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        number = calibri.render(str(x), False, (0, 0, 0))
        pygame.draw.line(graph2, (140, 140, 140), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph2, (0, 0, 0), (0, y), ( 5, y), 3)
        graph2.blit(number, (6, y - 5))
    points = \
        np.dstack([np.arange(0, GRAPH_WIDTH, GRAPH_WIDTH / 100) * 1.01, GRAPH_HEIGHT - (stats[3] / limit * GRAPH_HEIGHT)])[0]
    pygame.draw.lines(graph2, (255, 0, 0), False, points, 1)
    #endregion
    # region graph3
    limit = numpy.max(stats[4]) * 1.1
    graph3 = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
    graph3.fill((200, 200, 200))
    for x in np.arange(0.5, limit, 0.5):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        pygame.draw.line(graph3, (175, 175, 175), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph3, (0, 0, 0), (0, y), (2, y), 2)
    for x in np.arange(2.5, limit, 2.5):
        y = GRAPH_HEIGHT - x / limit * GRAPH_HEIGHT
        number = calibri.render(str(x), False, (0, 0, 0))
        pygame.draw.line(graph3, (140, 140, 140), (0, y), (GRAPH_WIDTH, y), 1)
        pygame.draw.line(graph3, (0, 0, 0), (0, y), ( 5, y), 3)
        graph3.blit(number, (6, y - 5))
    points = \
        np.dstack(
            [np.arange(0, GRAPH_WIDTH, GRAPH_WIDTH / 100) * 1.01, GRAPH_HEIGHT - (stats[4] / limit * GRAPH_HEIGHT)])[0]
    pygame.draw.lines(graph3, (255, 0, 0), False, points, 1)

    #endregion

    for x in range(50):
        foods.append(Food(random.randint(80, SCREEN_WIDTH - 80), random.randint(80, SCREEN_HEIGHT - 80)))

    clock = pygame.time.Clock()

    running = True
    while running and len(foods) > 5:
        screen.fill((20, 20, 20))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
                break

        for food in foods:
            food.draw()

        alive = [agent for agent in agents if agent.active]
        moving = [agent for agent in alive if agent.energy > 0]

        if len(moving) < len(agents) / 2:
            running = False
            break

        ray = raycast(foods, alive)

        for x, agent in enumerate(alive):
            if agent.energy <= 0:
                continue

            output = nets[agents.index(agent)].activate(ray[x])
            agent.move(output[0] * agent.speed)
            agent.turn(output[1] * agent.speed)

            agent.energy -= agent.size ** 3 * agent.speed ** 2 + agent.range

            for food in foods:
                if np.linalg.norm(food.position - agent.position) < agent.size + 8:
                    foods.remove(food)
                    ge[x].fitness += 1
                    agent.energy += ENERGY
                    if ge[x].fitness >= agent.need * 4:
                        agent.active = False

            for subject in agents:
                if np.linalg.norm(subject.position - agent.position) < (agent.size + subject.size) * SIZE_FACTOR:
                    if agent.size * 0.8 > subject.size and subject.active:
                        subject.dead = True,
                        subject.active = False
                        ge[x].fitness += 1
                        agent.energy += (ENERGY + subject.energy) * (subject.size / agent.size)
                        if ge[x].fitness >= agent.need * 4:
                            agent.active = False


        pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(0, 0, DATA_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(GRAPH_BORDER - GRAPH_OUTLINE, GRAPH_BORDER - GRAPH_OUTLINE, GRAPH_WIDTH + GRAPH_OUTLINE * 2, GRAPH_HEIGHT * 6 + GRAPH_OUTLINE * 10))

        screen.blit(graph1, (GRAPH_BORDER, GRAPH_BORDER))
        screen.blit(graph2, (GRAPH_BORDER, GRAPH_BORDER * 2 + GRAPH_HEIGHT))
        screen.blit(graph3, (GRAPH_BORDER, GRAPH_BORDER * 3 + GRAPH_HEIGHT*2))
        graph4 = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT * 3 + GRAPH_BORDER))
        graph4.fill((200, 200, 200))
        HEIGHT = GRAPH_HEIGHT * 3 + GRAPH_BORDER
        for i in np.arange(0.25, 5, 0.25):
            y = (5 - i) * HEIGHT/5
            x = i / 5 * GRAPH_WIDTH

            pygame.draw.line(graph4, (175, 175, 175), (0, y), (GRAPH_WIDTH, y), 1)
            pygame.draw.line(graph4, (0, 0, 0), (0, y), (2, y), 2)

            pygame.draw.line(graph4, (175, 175, 175), (x, 0), (x,HEIGHT), 1)
            pygame.draw.line(graph4, (0, 0, 0), (x, HEIGHT), (x, HEIGHT - 2), 2)

        for i in np.arange(1, 5, 1):
            y = (5 - i) * HEIGHT/5
            x = i / 5 * GRAPH_WIDTH

            number = calibri.render(str(i), False, (0, 0, 0))
            pygame.draw.line(graph4, (140, 140, 140), (0, y), (GRAPH_WIDTH, y), 1)
            pygame.draw.line(graph4, (0, 0, 0), (0, y), (5, y), 3)
            graph4.blit(number, (6, y - 5))
            
            pygame.draw.line(graph4, (140, 140, 140), (x, HEIGHT), (x, HEIGHT), 1)
            pygame.draw.line(graph4, (0, 0, 0), (x, HEIGHT), (x, HEIGHT-5), 3)
            graph4.blit(number, (x - 4, HEIGHT - 20))

        for subject in agents:
            pygame.draw.circle(graph4, (subject.range * 51, subject.speed * 51, subject.size*51),(subject.speed / 5 * GRAPH_WIDTH, (1-subject.range/5) * HEIGHT), subject.size*4)
        for agent in agents:
            agent.draw()
        screen.blit(graph4, (GRAPH_BORDER, GRAPH_BORDER * 4 + GRAPH_HEIGHT * 3))
        clock.tick(100)
        pygame.display.flip()
        pygame.display.update()

    for x, agent in enumerate(agents):
        if not agent.dead:
            ge[x].fitness = (ge[x].fitness / agent.need)
            if agent.energy / ENERGY > 1:
                ge[x].fitness *= (agent.energy / ENERGY)
        else:
            ge[x].fitness -= 2

    stats[0][0] = np.mean([x.range for x in ge])
    stats[1][0] = np.mean([x.radius for x in ge])
    stats[2][0] = np.mean([x.speed for x in ge])
    stats[3][0] = len(ge)
    stats[4][0] = np.mean([x.size() for x in ge])

    stats = np.roll(stats, -1, axis=1)



    #elif pop < 10:
    #   ge.sort(reverse=False, key=lambda x: x.fitness)
    #    for x in ge[next:]:
    #        x.fitness = 3
    #elif pop < 20:
    #    ge.sort(reverse=False, key=lambda x: x.fitness)
    #    for x in ge[next:]:
    #        x.fitness = 2




if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'originalconfig')
    run(config_path)
