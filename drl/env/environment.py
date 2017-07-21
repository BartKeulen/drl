from abc import abstractmethod, ABCMeta
import gym


class Environment(metaclass=ABCMeta):

    def __init__(self, name):
        self.name = name
        self.observation_space = None
        self.action_space = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def render(self, close=False):
        pass


class GymEnv(Environment):

    def __init__(self, name):
        super(GymEnv, self).__init__("Gym" + name)
        self.env = gym.make(name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)
