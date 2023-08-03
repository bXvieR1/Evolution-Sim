import numpy
import pygame
import random
import os
import neat
import numpy as np
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox


import reproduction

RAY_COUNT = 16
FOOD_RADIUS = 70
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 800
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
foodSlider = Slider(screen, 100, 100, 800, 400, min=0, max=100, step=1, initial=70)
foodText = TextBox(screen, 100, 100, 800, 400, fontSize=30)
gen = 0


class Agent():
    rotation = 90
    energy = 20
    def __init__(self, x, y, size):
        self.position = np.array([x, y])
        self.size = size/2

    def move(self, input):
        input *= 5
        self.position = np.add(self.position,
                               (np.sin(self.rotation) * (input / 2 + 0.25), np.cos(self.rotation) * (input / 2 + 0.25))) \
            .clip((10, 10), (SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10))

    def turn(self, input):
        self.rotation += 10 * (input / 180)

    def draw(self):
        pygame.draw.circle(screen, (255, 255, 255), (self.position[0], self.position[1]), self.size)
        pygame.draw.circle(screen, (0, 0, 180), (self.position[0] + np.sin(self.rotation + 0.6) * self.size,
                                                 self.position[1] + np.cos(self.rotation + 0.6) * self.size),
                                                 self.size / 2)
        pygame.draw.circle(screen, (0, 0, 180), (self.position[0] + np.sin(self.rotation - 0.6) * self.size,
                                                 self.position[1] + np.cos(self.rotation - 0.6) * self.size),
                                                 self.size / 2)
def raycast(foods, agents):

    output = np.empty((0, RAY_COUNT*2), int)
    positions = np.vstack(([food.position for food in foods], [agent.position for agent in agents]))
    angles = np.arange(1, RAY_COUNT + 1) - (RAY_COUNT + 1) / 2
    r = np.concatenate((np.full(len(foods), 8).astype(int), [agent.size for agent in agents]))

    for subject in agents:

        d = np.column_stack((np.sin(subject.rotation + angles * 0.1), np.cos(subject.rotation + angles * 0.1))) * 200
        f = subject.position - positions
        a = np.sum(d ** 2, axis=1)

        valid_food_indices = np.where(np.linalg.norm(f, axis=1) < FOOD_RADIUS)[0]

        c_values = np.sum(f[valid_food_indices] ** 2, axis=1) - r[valid_food_indices] ** 2
        b_values = 2 * np.dot(f[valid_food_indices], d.T)
        t_values = np.where(valid_food_indices < len(foods), -1, 1)
        discriminant_values = b_values ** 2 - 4 * a * c_values[:, np.newaxis]

        valid_ray_indices = np.where(discriminant_values >= 0)

        t1_values = (-b_values[valid_ray_indices] - np.sqrt(discriminant_values[valid_ray_indices])) / (
                2 * a[valid_ray_indices[1]])
        valid_t1_indices = (t1_values > 0) & (t1_values < 1)

        ray = np.ones(RAY_COUNT)
        hit = np.zeros(RAY_COUNT)

        ray[valid_ray_indices[1][valid_t1_indices]] = t1_values[valid_t1_indices]
        hit[valid_ray_indices[1][valid_t1_indices]] = t_values[valid_ray_indices[0][valid_t1_indices]]

        #for val in d:
        #    pygame.draw.line(screen, (180, 180, 180), subject.position, subject.position + val, 1)

        #if pygame.key.get_pressed()[pygame.K_DOWN] == 1:
        for val, t in zip(valid_ray_indices[1][valid_t1_indices], hit[valid_ray_indices[1][valid_t1_indices]]):
            color = (255, 0, 0) if t == 1 else (0, 255, 0) if t == -1 else (0, 0, 255)
            pygame.draw.line(screen, color, subject.position, subject.position + d[val] * ray[val], 1)
        output = np.vstack([output, np.concatenate((ray, hit))])

    return output

class Food():
    def __init__(self, x, y):
        self.position = np.array([x, y])
        # super(Food, self).__init__()

    def draw(self):
        pygame.draw.circle(screen, (0, 180, 0), (self.position[0], self.position[1]), 8)


def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        reproduction.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file)
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint(os.path.join(local_dir, "neat-checkpoint-24"))
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
        size = random.randint(8, 20)
        match genome_id % 4:
            case 0:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), 0, size))
            case 1:
                agents.append(Agent(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT, size))
            case 2:
                agents.append(Agent(0, random.randint(0, SCREEN_HEIGHT), size))
            case 3:
                agents.append(Agent(SCREEN_WIDTH, random.randint(0, SCREEN_HEIGHT), size))

        ge.append(genome)
    for x in range(foodSlider.getValue()):
        foods.append(Food(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)))

    clock = pygame.time.Clock()

    running = True
    while running and len(foods) > 10:
        screen.fill((0, 0, 0))
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
                agent.move(output[0])
                agent.turn(output[1])
                #agent.energy =- np.abs(output[0]) + 0.1
                for food in foods:
                    if np.linalg.norm(food.position-agent.position) < agent.size:
                        foods.remove(food)
                        ge[x].fitness = 1
                agent.draw()

        foodText.setText(foodSlider.getValue())

        clock.tick(100)
        pygame.display.flip()
        pygame.display.update()
        #print(clock.get_fps())


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
