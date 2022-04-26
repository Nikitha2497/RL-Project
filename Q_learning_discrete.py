import numpy as np
from policy import Policy
from typing import Tuple

from metric import Metric

def QLearning(
    env, 
    gamma:float, 
    alpha:float, 
    num_episode:int,
    eta: float,
    goal_reward: float,
    initQ:np.array,
    epsilon=.0,
    use_doubleQ=False) -> Tuple[np.array, Policy, Metric]:
    
    Q = initQ
    Q1 = initQ
    Q2 = initQ
    pi_star = GreedyPolicy(Q.shape[0], env.nS_columns)
    V_star_start = np.zeros(num_episode)
    Q_W_start = np.zeros(num_episode)
    Q_E_start = np.zeros(num_episode)
    a_star_start = np.zeros(num_episode)
    num_steps = np.zeros(num_episode)


    metric = Metric(num_episode)

    def epsilon_greedy_policy(s,epsilon=.0):
        nA = env.nA 
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            if use_doubleQ:
                return np.argmax(Q1[s] + Q2[s])
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

            #This is for double Q learning
            if use_doubleQ:
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
                    
                #this is an unsafe state
                elif done:
                    if is_Q1:
                        Q1[state][action] = Q1[state][action] + alpha*(reward - eta - Q1[state][action])
                    else:
                        Q2[state][action] = Q2[state][action] + alpha*(reward - eta - Q2[state][action])
                    break;
                
                #this is a safe state that is not a goal state
                if is_Q1:
                    Q1[state][action] = Q1[state][action] + alpha*(reward + gamma*Q1[new_state, np.argmax(Q2[new_state])] - Q1[state][action])
                else:
                    Q2[state][action] = Q2[state][action] + alpha*(reward + gamma*Q2[new_state, np.argmax(Q1[new_state])] - Q2[state][action])

            #This is for Q learning
            else:            
                #This is the goal state
                if done and goal:
                    Q[state][action] = Q[state][action] + alpha*(reward + goal_reward + gamma*max(Q[new_state]) - Q[state][action])
                    break;
                    
                #this is an unsafe state
                elif done:
                    Q[state][action] = Q[state][action] + alpha*(reward - eta + gamma*max(Q[new_state]) - Q[state][action])
                    break;
                
                #this is a safe state that is not a goal state
                Q[state][action] = Q[state][action] + alpha*(reward + gamma*max(Q[new_state]) - Q[state][action])
            state = new_state

        metric.set_v_star_start(i, np.max(Q[env.reset()]))
        metric.set_q_star_start(1, i, Q[env.reset()][1])
        metric.set_q_star_start(3, i , Q[env.reset()][3])
        metric.set_a_star_start(i,np.argmax(Q[env.reset()]))
        metric.set_num_steps(i, time_steps)
        
    for state in range(0, Q.shape[0]):
        pi_star.set_action(state, np.argmax(Q[state]))

    
    if use_doubleQ:
        return Q1, pi_star, metric
    
    return Q, pi_star, metric

class GreedyPolicy(Policy):

    def __init__(self, states:int, env_columns:int):
        self.env_columns = env_columns
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
        n = self.env_columns
        for state in self.state_action_dict:
        	if self.state_action_dict[state] == 0:
        		print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "N")

        	elif self.state_action_dict[state] == 1:
        		print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "W")

        	elif self.state_action_dict[state] == 2:
        		print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "S")

        	else:
        		print("state " , int(state/n) , int(state%n), " : " , self.state_action_dict[state], "E")
