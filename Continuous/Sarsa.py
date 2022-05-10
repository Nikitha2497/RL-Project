import numpy as np
import math
from policy import Policy
from features import StateActionFeatureVector
from typing import Tuple
from metric import Metric

#Implements the modified semi gradient sarsa algorithm
def Sarsa(
   env,  
    gamma:float, # discount factor
    alpha:float, # step size
    X:StateActionFeatureVector,
    num_episode:int,
    eta: float,
    goal_reward: float,
    epsilon: float
) -> Tuple[np.array, Policy, Metric]:
    
    metric = Metric(num_episode)
    w = np.zeros((X.feature_vector_len()))
    
    def epsilon_greedy_policy(s,w,epsilon=.0):
        nA = env.nA 
        Q = [np.dot(w, X(s,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
#             print("I have taken random action")
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    itr = 1; #to decay epsilon and alpha
    
    for i in range(0, num_episode):
#         print("==============================================")
#         print("episode", i)
        
        state = env.reset()
        action = epsilon_greedy_policy(state, w, epsilon)
        
        if (i%(num_episode/100)==0):
            epsilon = 1/(itr+1)
            alpha = 1/(itr+1)
            itr += 1
        
        while True:
            x = X(state, action) #feature vector
            
            q_hat = np.dot(w, x) #approximate state action value function
            
            new_state, reward, done, goal = env.step(action)
#             print("action", action, "state", new_state)
            
            #Goal state
            if done and goal:
                w = w + alpha*(reward + goal_reward - q_hat)*x
                break
            #Unsafe state
            elif done:
                w = w + alpha*(reward - eta - q_hat)*x
                break
            #Safe state
            else:
                new_action = epsilon_greedy_policy(new_state, w, epsilon)
    
                new_x = X(new_state, new_action)
                new_q_hat = np.dot(w, new_x)
                
                w = w + alpha*(reward + gamma*new_q_hat - q_hat)*x
                
                state = new_state
                action = new_action
            
        Q_start = [np.dot(w, X(env.reset(), a)) for a in range(env.nA)]
        
        metric.set_v_star_start(i, np.max(Q_start))
#         metric.set_q_star_start(0, i, Q_start[0])
        metric.set_q_star_start(1, i, Q_start[1])
#         metric.set_q_star_start(2, i, Q_start[2])
        metric.set_q_star_start(3, i, Q_start[3])
        
#             metric.set_a_star_start(i,np.argmax(Q_start))

    pi_star = GreedyPolicy(env.nA, w, X)
    
    #Prints the trajectory for the pi_star policy
    # state = env.reset()
    # while True:
    #     action = pi_star.action(state)
    #     new_state, reward, done, goal = env.step(action, False)
    #     print("action", action, "state", new_state)
         
    #     if done or goal:
    #         break
    #     else:
    #         state = new_state
        
    return w, pi_star, metric


class GreedyPolicy(Policy):

    def __init__(self, nA: int, w:np.array, X:StateActionFeatureVector):
        self.nA = nA
        self.w = w
        self.X = X

    def action_prob(self,state,action:int):
        Q = [np.dot(self.w, self.X(state,a)) for a in range(self.nA)]
        if action == np.argmax(Q):
            return 1.0
        return 0.0;

    def action(self,state):
        Q = [np.dot(self.w, self.X(state,a)) for a in range(self.nA)]
        return np.argmax(Q)

    def save_tofile(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.w)
            

    