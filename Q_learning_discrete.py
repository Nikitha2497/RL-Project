import numpy as np
from policy import Policy
from typing import Tuple

def QLearning(
    env, 
    gamma:float, 
    alpha:float, 
    num_episode:int,
    eta: float,
    goal_reward: float,
    initQ:np.array,
    epsilon=.0) -> Tuple[np.array, Policy, np.array, np.array, np.array, np.array, np.array]:
    
    Q = initQ
    pi_star = GreedyPolicy(Q.shape[0])
    V_star_start = np.zeros(num_episode)
    Q_W_start = np.zeros(num_episode)
    Q_E_start = np.zeros(num_episode)
    a_star_start = np.zeros(num_episode)
    num_steps = np.zeros(num_episode)

    def epsilon_greedy_policy(s,epsilon=.0):
        nA = env.nA 
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q[s])
    
    itr = 1; #to decay epsilon and alpha
    
    for i in range(0, num_episode):
        state = env.reset()
        time_steps = 0
        
        if (i%(num_episode/100)==0):
            epsilon = 1./(itr)
            alpha = 1./(itr)
            itr += 1
#             print("epsilon=", epsilon)
#             print("alpha=", alpha)
#             print("i=", i)
        
        while True:
            time_steps += 1
            action = epsilon_greedy_policy(state, epsilon)
            # print(action, "action")
        
            new_state, reward, done, goal = env.step(action)
            
            #This is the goal state
            if done and goal:
                Q[state][action] = Q[state][action] + alpha*(reward + goal_reward + gamma*max(Q[new_state]) - Q[state][action])
                break;
                
            #this is an unsafe state
            elif done:
                Q[state][action] = Q[state][action] + alpha*(reward - eta + gamma*max(Q[new_state]) - Q[state][action])
                break;
            
            #this is a safe state that is not a goal state
            else:
                Q[state][action] = Q[state][action] + alpha*(reward + gamma*max(Q[new_state]) - Q[state][action])
                state = new_state
        
        V_star_start[i] = np.max(Q[env.reset()])
        Q_W_start[i] = Q[env.reset()][1]
        Q_E_start[i] = Q[env.reset()][3]
        a_star_start[i] = np.argmax(Q[env.reset()])
        num_steps[i] = time_steps
        
    for state in range(0, Q.shape[0]):
        pi_star.set_action(state, np.argmax(Q[state]))
    
    return Q, pi_star, V_star_start, Q_W_start, Q_E_start, a_star_start, num_steps

class GreedyPolicy(Policy):

    def __init__(self, states:int):
        self.state_action_dict = {};
        for state in range(0,states):
            self.state_action_dict[state] = 1;

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
        		print("state " , int(state/8) , int(state%8), " : " , self.state_action_dict[state], "N")

        	elif self.state_action_dict[state] == 1:
        		print("state " , int(state/8) , int(state%8), " : " , self.state_action_dict[state], "W")

        	elif self.state_action_dict[state] == 2:
        		print("state " , int(state/8) , int(state%8), " : " , self.state_action_dict[state], "S")

        	else:
        		print("state " , int(state/8) , int(state%8), " : " , self.state_action_dict[state], "E")
