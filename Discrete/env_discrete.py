import numpy as np
from env import Env


class DiscreteEnv(Env): # MDP introduced at Fig 5.4 in Sutton Book

    def __init__(self,lambda1, lambda2, 
        primary_prob, 
        state_matrix):

        self.state_matrix = state_matrix

        self.m = self.state_matrix.shape[0]
        self.n = self.state_matrix.shape[1]

        self.__init_states()

        #Number of actions are always 4 for the discrete env
        self._nA = 4
        self._nS = self.m*self.n


        self._state = self.start_state

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.prob_action = {}
        #Actions
        #0 - North. 1 - West, 2 - South, 3 - East 
        
        #States
        #0 - North, 1- NorthWest, 2 - West, 3- SouthWest, 4- South
        #5- SouthEast, 6- East, 7 - NorthEast

        secondary_prob = (1 - primary_prob)/2

        self.prob_action[0] = [primary_prob, secondary_prob, 0, 0, 0, 0, 0 ,secondary_prob]

        self.prob_action[1] = [0, secondary_prob, primary_prob, secondary_prob, 0, 0, 0, 0]

        self.prob_action[2] = [0, 0, 0, secondary_prob, primary_prob, secondary_prob, 0, 0]

        self.prob_action[3] = [0, 0, 0, 0, 0, secondary_prob, primary_prob, secondary_prob]

    def __init_states(self):
        for i in range(0,self.m):
            for j in range(0,self.n):
                if self.state_matrix[i][j] == 2:
                    self.start_state = i*self.n + j
                elif self.state_matrix[i][j] == 1:
                    self.final_state = i*self.n + j 
                    self.final_state_i = i
                    self.final_state_j = j

    
    @property
    def nA(self) -> int:
        """ # possible actions """
        return self._nA

    @property
    def nS(self) -> int:
        """ # possible states """
        return self._nS

    @property
    def nS_rows(self) -> int:
        """ # possible states """
        return self.m

    @property
    def nS_columns(self) -> int:
        """ # possible states """
        return self.n



    def reset(self):
        # state = np.random.choice(int(self.m*self.n))
        # row = int(state / self.m)
        # col = int(state)%(self.m)
        
        # while self.state_matrix[row][col] != 0:
        #     state = np.random.choice(int(self.m*self.n))

        #     row = int(state / self.m)
        #     col = int(state)%(self.m)

        self._state = self.start_state
        return self._state

    def step(self, action, allow_noise=True):
        assert action in list(range(self._nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state

        if allow_noise:
            choice = np.random.choice(8, 1, p=self.prob_action[action])
        else:
            choice = action

        row = int(prev_state / self.n)
        col = int(prev_state)%(self.n)

        #North or North West or North East
        if choice == 0 or choice == 1 or choice == 7:
            if row > 0:
                row = row - 1

        #South or South West or South East
        if choice == 3 or choice == 4 or choice == 5:
            if row < self.m -1:
                row = row + 1
        
        #West or  North West or South West
        if choice == 1 or choice == 2 or choice == 3:
            if col > 0:
                col = col - 1;

        #East or South East or North East
        if choice == 5 or choice == 6 or choice == 7:
            if col < self.n - 1:
                col = col + 1

        self._state = row*self.m + col

        dist = abs(row - self.final_state_i) + abs(col - self.final_state_j)
        reward = -1*self.lambda1 - self.lambda2*dist

        #This is the state for terminal state but not goal state
        if self.state_matrix[row][col] == -1:
            return self._state, reward,  True, False 

        #This is the goal state
        elif self.state_matrix[row][col] == 1:
            return self._state, reward, True, True 

        #Remaining states
        return self._state, reward, False, False