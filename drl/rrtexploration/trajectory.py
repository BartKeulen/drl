from collections import deque
import numpy as np


class Trajectory(object):

    def __init__(self, states=None, actions=None):
        self.states = []
        self.actions = []
        self.length = 0

        if states is not None and actions is not None:
            self.states = [state for state in states]
            self.actions = [action for action in actions]
            self.length = len(self.states)

    def add_node(self, s, a):
        self.states.append(np.array(s))
        self.actions.append(np.array(a))
        self.length += 1

    def size(self):
        return self.length

    def get_states(self):
        return np.array(self.states)

    def get_actions(self):
        return np.array(self.actions)

    def split(self, split_ind):
        if split_ind < 1 or split_ind >= self.length:
            print("Invalid split index, must between 1 and size of trajectory. \
            [split_ind, size] = [%d, %d]" % (split_ind, self.size()))
            raise

        states = self.get_states()
        actions = self.get_actions()

        states = np.split(states, [split_ind])
        actions = np.split(actions, [split_ind])

        return Trajectory(states[0], actions[0]), Trajectory(states[1], actions[1])

    def reset(self):
        self.states = []
        self.actions = []
        self.length = 0
