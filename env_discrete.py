import numpy as np

from env import Env

class DiscreteEnv(Env): # MDP introduced at Fig 5.4 in Sutton Book

    def __init__(self):
        self.state_matrix = np.array([[-1,-1,-1,-1,-1,-1,-1],
                                       [-1,0,0,0,0,0,-1],
                                       [-1,0,-1,-1,0,0,-1],
                                       [-1,0,-1,-1,0,0,-1],
                                       [-1,0,1,0,0,0,-1],
                                       [-1,0,0,0,0,0,-1],
                                       [-1,-1,-1,-1,-1,-1,-1]]) 

        self.start_state = 9
        self._nA = 4
        self._state = self.start_state
        self.m = 7
        self.n = 7

        

    @property
    def nA(self) -> int:
        """ # possible actions """
        return self._nA

    def reset(self):
        self._state = self.start_state
        return self._state

    def step(self, action):
        assert action in list(range(self._nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        choice = np.random.choice(4)

        row = prev_state/self.m 
        col = prev_state%(self.m)

        if choice == 0:
            if row > 0:
                row = row - 1
        elif choice == 1:
            if col < n - 1:
                col = col + 1

        elif choice == 2:
            if row < m -1:
                row = row + 1
        else:
            if col > 0:
                col = col - 1;

        self._state = row*m + col

        if self.state_matrix[row][col] == -1:
            return self._state, -2, True, False 
        elif self.state_matrix[row][col] == 1:
            return self._state, 1, True, True 

        return self._state, -1, False, False