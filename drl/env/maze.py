from copy import deepcopy

from Box2D import *
import numpy as np
from gym import spaces

import pygame
from pygame.locals import *
import pygame.surfarray as surfarray

from drl.env.environment import Environment
from drl.utilities.numerical import rk_step


class Maze(Environment):
    WALL_HEIGHT = 4
    WALL_WIDTH = 4

    FORCE_SCALE = 150.
    PPM = 15.0
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS

    SIMPLE = 0
    MEDIUM = 1
    COMPLEX = 2

    def __init__(self, name, maze_structure):
        super(Maze, self).__init__(name)
        height = len(maze_structure) * Maze.WALL_HEIGHT + 4
        width = len(maze_structure[0]) * Maze.WALL_WIDTH + 4
        self._world_size = np.array([width, height])

        wall_pos = []
        self._init_pos, self._goal = None, None
        for i in range(len(maze_structure)):
            for j in range(len(maze_structure[0])):
                pos = (j * Maze.WALL_WIDTH + Maze.WALL_WIDTH/2 + 2, height - i * Maze.WALL_HEIGHT - Maze.WALL_HEIGHT/2 - 2)
                if maze_structure[i][j] == 1:
                    wall_pos.append(pos)
                elif maze_structure[i][j] == 2:
                    self._init_pos = b2Vec2(pos)
                elif maze_structure[i][j] == 3:
                    self._goal = b2Vec2(pos)
                elif maze_structure[i][j] > 3:
                    raise NotImplementedError("Teleports are not implemented yet.")

        if self._init_pos is None:
            raise Exception("Please specify an initial position in the maze structure using the number 2.")

        self._world = b2World(gravity=(0, 0))
        self._walls = []
        self._walls.append(self._world.CreateStaticBody(
            position=(width/2, 1),
            shapes=b2PolygonShape(box=(width/2, 1))
        ))
        self._walls.append(self._world.CreateStaticBody(
            position=(width/2, height - 1),
            shapes=b2PolygonShape(box=(width/2, 1))
        ))
        self._walls.append(self._world.CreateStaticBody(
            position=(1, height/2),
            shapes=b2PolygonShape(box=(1, height/2))
        ))
        self._walls.append(self._world.CreateStaticBody(
            position=(width - 1, height/2),
            shapes=b2PolygonShape(box=(1, height/2))
        ))

        for i in range(len(wall_pos)):
            wall = self._world.CreateStaticBody(
                position=wall_pos[i],
                shapes=b2PolygonShape(box=(Maze.WALL_WIDTH/2, Maze.WALL_HEIGHT/2))
            )
            self._walls.append(wall)

        self._body = self._world.CreateDynamicBody(position=self._init_pos,
                                                   linearDamping=0.5,
                                                   angularDamping=0.)
        box = self._body.CreateFixture(shape=b2CircleShape(pos=(0, 0), radius=Maze.WALL_HEIGHT/4),
                                       density=1,
                                       friction=0.,
                                       restitution=0.)

        self._u_high = np.ones(2)
        self._max_speed = np.sqrt(50)
        obs_high = np.ones(4)

        self._screen = None

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high)
        self.action_space = spaces.Box(low=-self._u_high, high=self._u_high)

        self.density_rgb_array = None
        self.initial_states = None
        self.trajectories = []

    def step(self, action):
        u = np.clip(action, -self._u_high, self._u_high)

        self._body.ApplyForce(force=u * Maze.FORCE_SCALE, point=self._body.position, wake=True)

        self._body.linearVelocity = np.clip(self._body.linearVelocity, -self._max_speed, self._max_speed)

        if self._goal is not None:
            done = (np.linalg.norm(self._body.position - self._goal) < Maze.WALL_HEIGHT/2)
        else:
            done = False
        reward = 1. if done else 0
        reward -= np.linalg.norm(action)*1e-4

        self._world.Step(Maze.TIME_STEP, 10, 10)

        return self._get_obs(), reward, done, {}

    def reset(self, x0=None):
        if x0 is None:
            self._body.position = b2Vec2(self._init_pos)
        else:
            x0 = np.multiply(x0, self._world_size/2) + self._world_size/2
            self._body.position = b2Vec2(x0)
        self._body.linearVelocity = b2Vec2(0, 0)
        self._body.angularVelocity = 0.
        self._body.angle = 0.
        return self._get_obs()

    def _get_obs(self, x=None):
        if x is None:
            x = np.concatenate((self._body.position, self._body.linearVelocity))
        pos = (np.asarray(x[:2]) - self._world_size/2) / (self._world_size/2)
        vel = np.asarray(x[2:]) / self._max_speed
        return np.concatenate((pos, vel))

    def get_goal(self):
        pos = (np.asarray(self._goal) - self._world_size/2) / (self._world_size/2)
        return np.pad(pos, (0, 2), 'constant')

    def _eom(self, x, u):
        A = np.zeros((4, 4))
        A[0, 2] = 1.
        A[1, 3] = 1

        B = np.zeros((4, 2))
        B[2, 0] = 1. / self._body.mass
        B[3, 1] = 1. / self._body.mass

        return np.dot(A, x) + np.dot(B, u)

    def dynamics_func(self, obs, u):
        x = np.zeros(4)
        x[:2] = obs[:2] * self._world_size / 2 + self._world_size / 2
        x[2:] = obs[2:] * self._max_speed
        u = np.clip(u, -self._u_high, self._u_high) * Maze.FORCE_SCALE

        xnew, xdot = rk_step(self._eom, x, u, Maze.TIME_STEP)

        return self._get_obs(xnew), None, None, None, None, None

    def set_density_rgb_array(self, density_rgb_array):
        self.density_rgb_array = density_rgb_array

    def clear_density_rgb_array(self):
        self.density_rgb_array = None

    def set_initial_states(self, initial_states):
        self.initial_states = initial_states

    def clear_samples(self):
        self.initial_states = None

    def add_trajectory(self, trajectory, color):
        self.trajectories.append((trajectory, color))

    def render(self, close=False):
        colors = {
            'background': (0, 0, 0, 0),
            b2_staticBody: (51, 107, 135, 255),
            b2_dynamicBody: (0, 255, 0, 255),
            'goal': (144, 175, 197, 255),
            'path': (255, 255, 255, 255)
        }

        if self._screen is not None:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    close = True

        if close:
            if self._screen is not None:
                pygame.display.quit()
                self._screen = None
            return

        if self._screen is None:
            self._screen = pygame.display.set_mode((int(self._world_size[0]*Maze.PPM), int(self._world_size[1]*Maze.PPM)), 0, 32)

            pygame.display.set_caption(self.name)
            self._clock = pygame.time.Clock()
            self.density_rgb_array = None
            self.initial_states = None

        self._screen.fill(colors['background'])

        if self.density_rgb_array is not None:
            surfarray.blit_array(self._screen, np.flip(self.density_rgb_array, 1))

        if self.initial_states is not None:
            for i in range(self.initial_states.shape[0]):
                pos = self.initial_states[i] * self._world_size / 2 + self._world_size / 2
                pos = pos * Maze.PPM
                pos = (int(pos[0]), int(self._world_size[1]*Maze.PPM - pos[1]))
                pygame.draw.circle(self._screen, (0, 255, 0, 255), pos, 2)

        if len(self.trajectories) > 0:
            for trajectory, color in self.trajectories:
                p_traj = []
                for node in trajectory:
                    pos = node[0][:2] * self._world_size / 2 + self._world_size/2
                    pos *= Maze.PPM
                    pos[1] = self._world_size[1]*Maze.PPM - pos[1]
                    p_traj.append(list(pos))
                pygame.draw.lines(self._screen, color, False, p_traj)

        # Draw walls
        for body in self._walls:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * Maze.PPM for v in shape.vertices]
                vertices = [(v[0], self._world_size[1]*Maze.PPM - v[1]) for v in vertices]
                pygame.draw.polygon(self._screen, colors[body.type], vertices)

        # Draw goal
        if self._goal is not None:
            pygame.draw.circle(self._screen, colors['goal'], (int(self._goal[0]*Maze.PPM),
                                                              int((self._world_size[1]-self._goal[1])*Maze.PPM)),
                               int(Maze.WALL_HEIGHT/2*Maze.PPM))

        # Draw agent
        pos = self._body.position * Maze.PPM
        pos = (int(pos[0]), int(self._world_size[1]*Maze.PPM - pos[1]))
        radius = int(self._body.fixtures[0].shape.radius * Maze.PPM)

        pygame.draw.circle(self._screen, colors[self._body.type], pos, radius)

        pygame.display.flip()
        self._clock.tick(Maze.TARGET_FPS)

    def save_frame(self, fp):
        if self._screen is not None:
            pygame.image.save(self._screen, fp)

    @staticmethod
    def generate_maze(type, goal=[1, 1]):
        if type == Maze.SIMPLE:
            name = "SimpleMaze"
            maze_structure = [[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]]
        elif type == Maze.MEDIUM:
            name = "MediumMaze"
            maze_structure = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        elif type == Maze.COMPLEX:
            name = "ComplexMaze"
            maze_structure = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        else:
            raise Exception("Please choose from the available maze types: Maze.{SIMPLE, MEDIUM, COMPLEX}")

        if goal is not None:
            maze_structure[goal[1]][goal[0]] = 3

        return Maze(name, maze_structure)

if __name__ == "__main__":

    maze = input("Choose which maze to render: [simple, medium, complex] ")

    if maze == 'simple':
        type = Maze.SIMPLE
    elif maze == 'medium':
        type = Maze.MEDIUM
    elif maze == 'complex':
        type = Maze.COMPLEX
    else:
        raise Exception("%s is not a valid maze, choose from available options." % maze)

    env = Maze.generate_maze(type)
    env.reset()
    env.render()

    running = True
    steps = 0
    while running:
        action = [0, 0]

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

            if event.type == KEYDOWN:
                if event.key == K_UP:
                    action = [0, 1]
                if event.key == K_DOWN:
                    action = [0, -1]
                if event.key == K_LEFT:
                    action = [-1, 0]
                if event.key == K_RIGHT:
                    action = [1, 0]

        obs, r, t, _ = env.step(action)
        steps += 1

        if t:
            obs = env.reset()
            print("Number of steps: %d" % steps)
            steps = 0

        print("Obs: %s, r: %.2f, t: %d" % (obs, r, t))

        env.render(close=(not running))