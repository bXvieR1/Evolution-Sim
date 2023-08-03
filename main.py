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
SCREEN_HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
foodSlider = Slider(screen, 100, 100, 800, 400, min=0, max=100, step=1, initial=70)
foodText = TextBox(screen, 100, 100, 800, 400, fontSize=30)
gen = 0


class Agent():
    rotation = 90
    energy = 20
    def __init__(self, x, y, genome):
        self.position = np.array([x, y])
        self.size = genome.radius * 5
        self.range = genome.range * 50
        self.speed = genome.speed
    def move(self, input):
        input *= 5
        self.position = np.add(self.position,
                               (np.sin(self.rotation) * (input / 2 + 0.25), np.cos(self.rotation) * (input / 2 + 0.25))) \
            .clip((10, 10), (SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10))

    def turn(self, input):
        self.rotation += 10 * (input / 180)

    def draw(self):

        body = numpy.clip((int)(self.speed * 50), 0, 50)
        eyes = numpy.clip(255 - (int)(self.range / 20), 0, 255)

        pygame.draw.circle(screen, (0, 0, 0), self.position,  self.size + 2)
        pygame.draw.circle(screen, (250 - body, 180, 200 + body), self.position, self.size)

        eye1 = (self.position[0] + np.sin(self.rotation + 0.6) * self.size, self.position[1] + np.cos(self.rotation + 0.6) * self.size)
        pygame.draw.circle(screen, (10, 10, 10), eye1, self.size / 2 + 1)
        pygame.draw.circle(screen, (255, eyes, eyes), eye1, self.size / 2)

        eye2 = (self.position[0] + np.sin(self.rotation - 0.6) * self.size, self.position[1] + np.cos(self.rotation - 0.6) * self.size)
        pygame.draw.circle(screen, (10, 10, 10), eye2, self.size / 2 + 1)
        pygame.draw.circle(screen, (255, eyes, eyes), eye2, self.size / 2)


def raycast(foods, agents):
    output = []
    positions = np.vstack(([food.position for food in foods], [agent.position for agent in agents]))
    angles = np.arange(1, RAY_COUNT + 1) - (RAY_COUNT + 1) / 2
    r = np.concatenate((np.full(len(foods), 4).astype(int), [agent.size for agent in agents]))

    for subject in agents:
        d = np.column_stack((np.sin(subject.rotation + angles * 0.1), np.cos(subject.rotation + angles * 0.1))) * subject.range
        f = subject.position - positions
        a = np.sum(d ** 2, axis=1)

        valid_food_indices = np.where(np.linalg.norm(f, axis=1) < subject.range)[0]

        c_values = np.sum(f[valid_food_indices] ** 2, axis=1) - r[valid_food_indices] ** 2
        b_values = 2 * np.dot(f[valid_food_indices], d.T)
        t_values = valid_food_indices < len(foods)
        discriminant_values = b_values ** 2 - 4 * a * c_values[:, np.newaxis]

        valid_ray_indices = np.where(discriminant_values >= 0)

        t1_values = (-b_values[valid_ray_indices] - np.sqrt(discriminant_values[valid_ray_indices])) / (2 * a[valid_ray_indices[1]])
        valid_t1_indices = (t1_values > 0) & (t1_values < 1)

        agentRay = np.zeros(RAY_COUNT)
        foodRay = np.zeros(RAY_COUNT)

        valid_ray_t1_indices = valid_ray_indices[1][valid_t1_indices]
        valid_ray_t1_values = t1_values[valid_t1_indices]

        agentRay[valid_ray_t1_indices] = valid_ray_t1_values * (~t_values[valid_ray_indices[0][valid_t1_indices]])
        foodRay[valid_ray_t1_indices] = valid_ray_t1_values * t_values[valid_ray_indices[0][valid_t1_indices]]

        for x, (f, a) in enumerate(zip(foodRay, agentRay)):
            if f == 0:
                pygame.draw.line(screen, (255, 0, 0), subject.position, subject.position + d[x] * a, 1)
            else:
                pygame.draw.line(screen, (0, 255, 0), subject.position, subject.position + d[x] * f, 1)

        output.append(np.concatenate([subject.range * foodRay, subject.range * agentRay]))
    return np.vstack(output)


class Food():
    def __init__(self, x, y):
        self.position = np.array([x, y])
        # super(Food, self).__init__()

    def draw(self):
        pygame.draw.circle(screen, (0, 180, 0), (self.position[0], self.position[1]), 4)


def run(config_file):
    config = neat.config.Config(
        genome.DefaultGenome,
        reproduction.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file)
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint(os.path.join(local_dir, "neat-checkpoint-19"))
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
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), 0, genome))
            case 1:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT, genome))
            case 2:
                agents.append(Agent(0, random.randint(0, SCREEN_HEIGHT), genome))
            case 3:
                agents.append(Agent(SCREEN_WIDTH, random.randint(0, SCREEN_HEIGHT), genome))

        ge.append(genome)
    for x in range(70):
        foods.append(Food(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)))

    clock = pygame.time.Clock()

    running = True
    while running and len(foods) > 20:
        screen.fill((20, 20, 20))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
                break

        for food in foods:
            food.draw()

        #print(pygame.mouse.get_pos())

        ray = raycast(foods, agents)

        for x, agent in enumerate(agents):

            if agent.energy > 0:
                output = nets[agents.index(agent)].activate(ray[x])
                agent.move(output[0] * agent.speed)
                agent.turn(output[1] * agent.speed)
                #agent.energy =- np.abs(output[0]) + 0.1
                for food in foods:
                    if np.linalg.norm(food.position-agent.position) < agent.size + 8:
                        foods.remove(food)
                        ge[x].fitness = 1
                agent.draw()

        #foodText.setText(foodSlider.getValue())

        clock.tick(100)
        pygame.display.flip()
        pygame.display.update()
        print(clock.get_fps())


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
