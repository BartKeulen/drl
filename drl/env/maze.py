from Box2D import *
import numpy as np
from gym import spaces

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_RIGHT, K_LEFT


class Maze(object):
    FORCE_SCALE = 50.
    TORQUE_SCALE = 100.
    PPM = 20.0
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS

    def __init__(self, world_size, wall_pos, wall_sizes, init_pos, goal):
        self._world_size = np.array(world_size)
        self._init_pos = init_pos

        self._world = b2World(gravity=(0, 0))
        self._walls = []

        assert len(wall_pos) == len(wall_sizes)
        for i in range(len(wall_pos)):
            wall = self._world.CreateStaticBody(
                position=wall_pos[i],
                shapes=b2PolygonShape(box=wall_sizes[i])
            )
            self._walls.append(wall)

        self._body = self._world.CreateDynamicBody(position=self._init_pos,
                                                   angle=np.pi * 3 / 2,
                                                   linearDamping=0.,
                                                   angularDamping=0.)
        box = self._body.CreateFixture(shape=b2CircleShape(pos=(0, 0), radius=2.),
                                       density=1,
                                       friction=0.,
                                       restitution=0.)

        self._goal = b2Vec2(goal)

        self._u_high = np.ones(2)
        self._max_speed = np.sqrt(50)
        obs_high = np.ones(2)

        self._screen = None

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high)
        self.action_space = spaces.Box(low=-self._u_high, high=self._u_high)

    def step(self, u):
        u = np.clip(u, -self._u_high, self._u_high)

        # self._body.angle += u[1] * Maze.ANGLE_SCALE * np.pi / 180.

        # self._body.ApplyTorque(u[1] * Maze.TORQUE_SCALE, wake=True)

        # action = u[0] * Maze.FORCE_SCALE
        # action = (action*np.cos(self._body.angle), action*np.sin(self._body.angle))
        self._body.ApplyForce(force=u * Maze.FORCE_SCALE, point=self._body.position, wake=True)

        self._body.linearVelocity = np.clip(self._body.linearVelocity, -self._max_speed, self._max_speed)

        terminal = (np.linalg.norm(self._body.position - self._goal) < 2.)
        # reward = 100. if terminal else -np.linalg.norm(u)/10.
        reward = 1. if terminal else 0

        return self._get_obs(), reward, terminal, {}

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

    def _get_obs(self):
        pos = np.array([self._body.position[0], self._body.position[1]])
        pos = (pos - self._world_size/2) / (self._world_size/2)
        return pos
        # goal = np.array([self._goal[0], self._goal[1]])
        # goal = (goal - self._world_size/2) / (self._world_size/2)
        # angle = self._body.angle
        # vel = np.linalg.norm(self._body.linearVelocity) / self._max_speed
        #
        # return np.array([pos[0], pos[1], np.cos(angle), np.sin(angle), vel, goal[0], goal[1]])

    def render(self, close=False):
        colors = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (127, 127, 127, 255),
            'goal': (0, 255, 0, 255)
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
            pygame.display.set_caption(self.__class__.__name__)
            self._clock = pygame.time.Clock()

        self._screen.fill((0, 0, 0, 0))

        # Draw walls
        for body in self._walls:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * Maze.PPM for v in shape.vertices]
                vertices = [(v[0], self._world_size[1]*Maze.PPM - v[1]) for v in vertices]
                pygame.draw.polygon(self._screen, colors[body.type], vertices)

        # Draw goal
        pygame.draw.circle(self._screen, colors['goal'], (int(self._goal[0]*Maze.PPM),
                                                          int((self._world_size[1]-self._goal[1])*Maze.PPM)),
                           int(2.*Maze.PPM))

        # Draw agent
        pos = self._body.position * Maze.PPM
        pos = (int(pos[0]), int(self._world_size[1]*Maze.PPM - pos[1]))
        radius = int(self._body.fixtures[0].shape.radius * Maze.PPM)

        pygame.draw.circle(self._screen, colors[self._body.type], pos, radius)
        # angle = self._body.angle
        # pos2 = (self._body.position + b2Vec2(4.*np.cos(angle), 4.*np.sin(angle))) * Maze.PPM
        # pos2 = (int(pos2[0]), int(self._world_size[1]*Maze.PPM - pos2[1]))
        # pygame.draw.line(self._screen, colors[self._body.type], pos, pos2, 2)

        self._world.Step(Maze.TIME_STEP, 10, 10)

        pygame.display.flip()
        self._clock.tick(Maze.TARGET_FPS)

    def check_point_intersects_wall(self, point):
        aabb = b2AABB(lowerBound=point-(0.001, 0.001), upperBound=point+(0.001, 0.001))

        query = Maze.CheckWallQueryCallback()
        self._world.QueryAABB(query, aabb)

        print(query.intersect_wall)

    class CheckWallQueryCallback(b2QueryCallback):

        def __init__(self):
            b2QueryCallback.__init__(self)
            self.intersect_wall = False

        def ReportFixture(self, fixture):
            if fixture.body.type == b2_staticBody:
                self.intersect_wall = True
            return True


class SimpleMaze(Maze):

    def __init__(self):
        world_size = (32, 32)
        wall_pos = [(16, 2), (16, 30), (2, 16), (30, 16), (9, 16)]
        wall_sizes = [(16, 2), (16, 2), (2, 16), (2, 16), (9, 2)]
        init_pos = (9, 9)
        goal = (9, 23)

        super(SimpleMaze, self).__init__(world_size, wall_pos, wall_sizes, init_pos, goal)


class MediumMaze(Maze):

    def __init__(self):
        world_size = (50, 50)
        wall_pos = [(25, 2), (25, 48), (2, 25), (48, 25), (18, 18), (32, 32)]
        wall_sizes = [(25, 2), (25, 2), (2, 25), (2, 25), (18, 2), (18, 2)]
        init_pos = (9, 9)
        goal = (41, 41)

        super(MediumMaze, self).__init__(world_size, wall_pos, wall_sizes, init_pos, goal)

if __name__ == "__main__":
    maze = input("Choose which maze to render: [simple, medium] ")

    if maze == 'simple':
        env = SimpleMaze()
    elif maze == 'medium':
        env = MediumMaze()
    else:
        raise Exception("%s is not a valid maze, choose from available options." % maze)

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
                    action = [1, 0]
                if event.key == K_DOWN:
                    action = [-1, 0]
                if event.key == K_LEFT:
                    action = [0, 1]
                if event.key == K_RIGHT:
                    action = [0, -1]

        obs, r, t, _ = env.step(action)
        steps += 1

        if t:
            obs = env.reset()
            print("Number of steps: %d" % steps)
            steps = 0

        # point_str = input("Give point to check: ")
        #
        # if point_str == "exit":
        #     running = False
        #     continue
        #
        # point_str = point_str.split(" ")
        # point = b2Vec2(float(point_str[0]), float(point_str[1]))
        #
        # env.check_point_intersects_wall(point)

        print("Obs: %s, r: %.2f, t: %d" % (obs, r, t))

        env.render(close=(not running))