import numpy as np
import math
from policy import Policy
from typing import Tuple

#This is for discrete state and action space

def QLearning(
    env, 
    gamma:float, 
    alpha:float, 
    num_episode:int,
    eta: float,
    initQ:np.array,
    epsilon=.0,
    epsilon_factor=100) -> Tuple[np.array,Policy]:
    
    Q = initQ
    pi_star = GreedyPolicy(Q.shape[0])

    def epsilon_greedy_policy(s,epsilon=.0):
        nA = env.nA 
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q[s])

    time_step = 0
    for i in range(0, num_episode):
        state = env.reset()
        # print("episode ", i , " start state: ",  state)    
       	 
        while True:
            time_step += 1
            alpha = 1./time_step
            epsilon = 1./(epsilon_factor*time_step)


            action = epsilon_greedy_policy(state, epsilon)
           
            new_state, reward, done, goal = env.step(action)
            

            #This is the goal state
            if done and goal:
                Q[state][action] = Q[state][action] + alpha*(reward - Q[state][action])
                break;

            if done:
                Q[state][action] = Q[state][action] + alpha*(reward - eta - Q[state][action])
                break;

            Q[state][action] = Q[state][action] + alpha*(reward + gamma*max(Q[new_state]) - Q[state][action])
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
        	if self.state_action_dict[state] == 0:
        		print("state " , int(state/7) , int(state%7), " : " , self.state_action_dict[state], "N")

        	elif self.state_action_dict[state] == 1:
        		print("state " , int(state/7) , int(state%7), " : " , self.state_action_dict[state], "W")

        	elif self.state_action_dict[state] == 2:
        		print("state " , int(state/7) , int(state%7), " : " , self.state_action_dict[state], "S")

        	else:
        		print("state " , int(state/7) , int(state%7), " : " , self.state_action_dict[state], "E")



            

