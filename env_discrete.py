import numpy as np


from env import Env

class DiscreteEnv(Env): # MDP introduced at Fig 5.4 in Sutton Book

    def __init__(self,lambda1, lambda2, primary_prob, secondary_prob):
        self.state_matrix = np.array([[-1,-1,-1,-1,-1,-1,-1],
                                       [-1,0,0,0,0,0,-1],
                                       [-1,0,-1,-1,0,0,-1],
                                       [-1,0,-1,-1,0,0,-1],
                                       [-1,0,1,0,0,0,-1],
                                       [-1,0,0,0,0,0,-1],
                                       [-1,-1,-1,-1,-1,-1,-1]]) 

        self.start_state = 9
        self._nA = 4
        self._nS = 49
        self._state = self.start_state
        self.m = 7
        self.n = 7
        self.final_state = 30


        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.final_state_i = int (self.final_state/self.m)
        self.final_state_j = int(self.final_state)%(self.m)

        self.prob_action = {}
        #0 - North. 1 - West, 2 - South, 3 - East - Actions

        #0 - North, 1- NorthWest, 2 - West, 3- SouthWest, 4- South
        #5- SouthEast, 6- East, 7 - NorthEast
        self.prob_action[0] = [primary_prob, secondary_prob, 0, 0, 0, 0, 0 ,secondary_prob]

        self.prob_action[1] = [0, secondary_prob, primary_prob, secondary_prob, 0, 0, 0, 0]

        self.prob_action[2] = [0, 0, 0, secondary_prob, primary_prob, secondary_prob, 0, 0]

        self.prob_action[3] = [0, 0, 0, 0, 0, secondary_prob, primary_prob, secondary_prob]
    
    @property
    def nA(self) -> int:
        """ # possible actions """
        return self._nA

    @property
    def nS(self) -> int:
        """ # possible states """
        return self._nS

    def reset(self):
        self._state = self.start_state
        return self._state

    def step(self, action):
        assert action in list(range(self._nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state

        choice = np.random.choice(8, 1, p=self.prob_action[action])

        row = int(prev_state / self.m)
        col = int(prev_state)%(self.m)

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
        cost = self.lambda1 + self.lambda2*dist

        #This is the state for terminal state but not goal state
        if self.state_matrix[row][col] == -1:
            return self._state, cost,  True, False 

        #This is the goal state
        elif self.state_matrix[row][col] == 1:
            return self._state, cost, True, True 

        #Remaining states
        return self._state, cost, False, False