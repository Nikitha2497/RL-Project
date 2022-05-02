import numpy as np


class Env(object):

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

    @property
    def nA(self) -> int:
        """ # possible actions """
        raise NotImplementedError()

    @property
    def nS(self) -> int:
        """ # possible states """
        raise NotImplementedError()

    @property
    def nS_rows(self) -> int:
        """ # possible rows only for discrete env """
        raise NotImplementedError()

    @property
    def nS_columns(self) -> int:
        """ # possible rows only for discrete env """
        raise NotImplementedError()