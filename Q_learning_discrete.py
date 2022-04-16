import numpy as np
import math
from policy import Policy

#This is for discrete state and action space

def QLearning(
    env, 
    gamma:float, # discount factor
    alpha:float, # step size
    num_episode:int,
    eta: float,
    epsilon=.0,
    initQ:np.array
) -> Tuple[np.array,Policy]::
	
	Q = initQ

	pi_star = GreedyPolicy(Q.)

	def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.nA #TODO - Define the environment in this way

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    for i in range(0, num_episode):
        state = env.reset()
        done = False
       	 
        while True:
            action = epsilon_greedy_policy(new_state, done, w, epsilon)
            new_state, reward, done, goal = env.step(action)

            #This is the goal state
			if done and goal:
				Q[state][action] = Q[state][action] + alpha*(reward - Q[state][action])
				break;

			if done:
				Q[state][action] = Q[state][action] + alpha*(reward + eta - Q[state][action])
				break;
			
			Q[state][action] = Q[state][action] + alpha*(reward + min(Q[state]) - Q[state][action])

			pi_star.set_action(state, np.argmax(Q[state]))

            state = new_state
    
    return Q, pi_star



class GreedyPolicy(Policy):

    def __init__(self, states:int):
        self.state_action_dict = {};
        for state in range(0,states):
            self.state_action_dict[state] = 0;

    def set_action(self,state:int, action:int):
        self.state_action_dict[state] = action;

    def action_prob(self,state:int,action:int):
        if(self.state_action_dict[state] == action):
            return 1.0;
        return 0.0;

    def action(self,state:int):
        return self.state_action_dict[state];

    def print_all(self):
        for state in self.state_action_dict:
            print("state " , state, " : " , self.state_action_dict[state])


