import numpy as np


class Env(object):
    def __init__(self):

    @property
    def spec(self) -> EnvSpec:
        return self._env_spec

    def reset(self) -> int:
        """
        reset the environment. It should be called when you want to generate a new episode
        return:
            initial state
        """
        raise NotImplementedError()

    def step(self,action:int) -> (int, int, bool, bool):
        """
        proceed one step.
        return:
            next state, reward, done (whether it reached to a terminal state), goal (whether it is the goal state or not)
        """
        raise NotImplementedError()