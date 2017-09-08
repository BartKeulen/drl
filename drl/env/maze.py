from copy import deepcopy
import os

from Box2D import *
import numpy as np
from gym.spaces import Box

import pygame
from pygame.locals import *
import pygame.surfarray as surfarray

from drl.env.environment import Environment


class CollisionDetector(b2ContactListener):

    def __init__(self):
        b2ContactListener.__init__(self)
        self.collision = False

    def BeginContact(self, contact):
        self.collision = True

    def EndContact(self, contact):
        self.collision = False


class Maze(Environment):
    WALL_HEIGHT = 4
    WALL_WIDTH = 4

    FORCE_SCALE = 50.
    PPM = 15.0
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS

    SIMPLE = 0
    MEDIUM = 1
    COMPLEX = 2
    EMPTY = 3

    def __init__(self, name, maze_structure, goal_reward=100., collision_penalty=0.001):
        super(Maze, self).__init__(name)
        height = len(maze_structure) * Maze.WALL_HEIGHT + 4
        width = len(maze_structure[0]) * Maze.WALL_WIDTH + 4
        self._world_size = np.array([width, height])

        self._goal_reward = goal_reward
        self._collision_penalty = collision_penalty

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

        self._collision_detector = CollisionDetector()
        self._world = b2World(gravity=(0, 0), contactListener=self._collision_detector)
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
                                       density=0.,
                                       friction=0.,
                                       restitution=0.)

        c = self._body.linearDamping
        m = self._body.mass
        dt = Maze.TIME_STEP

        A = np.eye(4)
        A[0, 2] = dt
        A[1, 3] = dt
        A[2, 2] -= c / m * dt
        A[3, 3] -= c / m * dt
        B = np.zeros((4, 2))
        B[2, 0] = Maze.FORCE_SCALE * dt / m
        B[3, 1] = Maze.FORCE_SCALE * dt / m
        self.A, self.B = A, B

        self._u_high = np.ones(2)
        self._max_speed = np.sqrt(50)
        obs_high = np.ones(4)

        self._screen = None

        self.observation_space = Box(low=-obs_high, high=obs_high)
        self.action_space = Box(low=-self._u_high, high=self._u_high)

        self.density_rgb_array = None
        self.initial_states = None
        self.trajectories = []

        self.rec_count = 0

    def step(self, u):
        u = np.clip(u, -self._u_high, self._u_high)

        # self._body.ApplyForce(force=u * Maze.FORCE_SCALE, point=self._body.position, wake=True)
        # self._body.linearVelocity = np.clip(self._body.linearVelocity, -self._max_speed, self._max_speed)

        x = np.concatenate((self._body.position, self._body.linearVelocity))
        x_new = self.dynamics(x, u)
        self._body.position = x_new[:2]
        self._body.linearVelocity = x_new[2:]

        if self._goal is not None:
            done = (np.linalg.norm(self._body.position - self._goal) < Maze.WALL_HEIGHT/2)
        else:
            done = False
        reward = self._goal_reward if done else 0
        reward -= self._collision_penalty if self._collision_detector.collision else 0
        reward -= np.linalg.norm(u) * 1e-4

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
        return x

    def get(self, t):
        return self.A, self.B

    def get_goal(self):
        pos = (np.asarray(self._goal) - self._world_size/2) / (self._world_size/2)
        return np.pad(pos, (0, 2), 'constant')

    def dynamics(self, x, u):
        u = np.clip(u, -self._u_high, self._u_high)

        x_new = self.A.dot(x) + self.B.dot(u)

        return x_new.copy()

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

    def render(self, close=False, record=False):
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

        self._screen.fill(colors['background'])

        if self.density_rgb_array is not None:
            surfarray.blit_array(self._screen, np.flip(self.density_rgb_array, 1))

        if self.initial_states is not None:
            pos = self.initial_states[:, :2].copy() * Maze.PPM

            for i in range(pos.shape[0]):
                pygame.draw.circle(self._screen, (0, 255, 0, 255),
                                   (int(pos[i, 0]), int(self._world_size[1]*Maze.PPM - pos[i, 1])), 2)

        if len(self.trajectories) > 0:
            for trajectory, color in self.trajectories:
                pos = trajectory[:, :2].copy()
                # pos = trajectory[:, :2] * self._world_size / 2 + self._world_size/2
                pos *= Maze.PPM
                pos[:, 1] = self._world_size[1]*Maze.PPM - pos[:, 1]
                pygame.draw.lines(self._screen, color, False, list(pos))

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

    def save_frame(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        fp = os.path.join(path, "frame_%.10d.png" % self.rec_count)
        self.rec_count += 1
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
            goal_reward = 100.
            collision_penalty = 0.001
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
            goal_reward = 100.
            collision_penalty = 0.0001
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
            goal_reward = 100.
            collision_penalty = 0.0001
        elif type == Maze.EMPTY:
            name = "ComplexMaze"
            maze_structure = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            goal_reward = 100.
            collision_penalty = 0.0001
        else:
            raise Exception("Please choose from the available maze types: Maze.{SIMPLE, MEDIUM, COMPLEX}")

        if goal is not None:
            maze_structure[goal[1]][goal[0]] = 3

        return Maze(name, maze_structure, goal_reward, collision_penalty)

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
    obs = env.reset()
    env.render()

    running = True
    steps = 0
    while running:
        action = [-0.2, 0]

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

        # Fm, fv, dyn_cov = env.dynamics()
        # obs_hat = np.dot(Fm, np.hstack((obs, np.asarray(action))).T)

        obs_dyn = env.dynamics(obs, action)

        obs_tp1, r, t, _ = env.step(action)
        steps += 1

        obs = obs_tp1
        if t:
            obs = env.reset()
            print("Number of steps: %d" % steps)
            steps = 0

        print("Obs: %s, Dyn: %s r: %.2f, t: %d" % (obs, obs_dyn, r, t))

        env.render(close=(not running))