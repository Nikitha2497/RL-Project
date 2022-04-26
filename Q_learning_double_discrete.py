import numpy as np
import math
from policy import Policy
from typing import Tuple

#This is for discrete state and action space

def DoubleQLearning(
    env, 
    gamma:float, 
    alpha:float, 
    num_episode:int,
    eta: float,
    goal_reward: float,
    initQ:np.array,
    epsilon=.0,
    epsilon_factor=100) -> Tuple[np.array,Policy, np.array, np.array, np.array]:
    
    Q1 = initQ
    Q2 = initQ
    pi_star = GreedyPolicy(Q1.shape[0])

    v_star_start = np.zeros(num_episode)
    Q_star_start_east = np.zeros(num_episode)
    a_star_start = np.zeros(num_episode)

    def epsilon_greedy_policy(s,epsilon=.0):
        nA = env.nA 
        if np.random.rand() < epsilon:
            # print("taken a random action")
            return np.random.randint(nA)
        else:
            return np.argmax(Q1[s] + Q2[s])

    time_step = 0   
    for i in range(0, num_episode):
        state = env.reset()
        start_state = state
        # print("episode ", i , " start state: ",  state)    
       	epsilon = 1./(epsilon_factor*(i+1))
        while True:
            time_step += 1
            alpha = 1./time_step
            # epsilon = 1./(epsilon_factor*time_step)

            # print("updated state", int(state/7) , int(state%7))
            action = epsilon_greedy_policy(state, epsilon)
           
            new_state, reward, done, goal = env.step(action)

            is_Q1 = False
            if np.random.rand() < 0.5:
                is_Q1 = True
            
            #This is the goal state
            if done and goal:
                if is_Q1:
                    Q1[state][action] = Q1[state][action] + alpha*(reward + goal_reward - Q1[state][action])
                else:
                    Q2[state][action] = Q2[state][action] + alpha*(reward  + goal_reward - Q2[state][action])
                break;

            if done:
                if is_Q1:
                    Q1[state][action] = Q1[state][action] + alpha*(reward - eta - Q1[state][action])
                else:
                    Q2[state][action] = Q2[state][action] + alpha*(reward - eta - Q2[state][action])
                break;

            if is_Q1:
                Q1[state][action] = Q1[state][action] + alpha*(reward + gamma*Q1[new_state, np.argmax(Q2[new_state])] - Q1[state][action])
            else:
                Q2[state][action] = Q2[state][action] + alpha*(reward + gamma*Q2[new_state, np.argmax(Q1[new_state])] - Q2[state][action])
            # Q[state][action] = Q[state][action] + alpha*(reward + gamma*max(Q[new_state]) - Q[state][action])
            

            state = new_state

        for state in range(0,Q1.shape[0]):
            pi_star.set_action(state, np.argmax(Q1[state]))
            # print(int(state/7) , int(state%7)," : ",  Q1[state])

        # v_star_start[i] = max(Q1[start_state])
        v_star_start[i] = Q1[start_state][1]
        Q_star_start_east[i] = Q1[start_state][3]
        a_star_start[i] = np.argmax(Q1[start_state])
    
    return Q1, pi_star, v_star_start, a_star_start, Q_star_start_east



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



            

