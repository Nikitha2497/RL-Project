import numpy as np
import math

from policy import Policy

from features import StateActionFeatureVector

from typing import Tuple
from env import Env


def Sarsa(
   env,  
    gamma:float, # discount factor
    alpha:float, # step size
    X:StateActionFeatureVector,
    num_episode:int,
    eta: float,
    goal_reward: float,
    epsilon=0.01
) -> Tuple[np.array, Policy]:
    """
    Implement True online Sarsa(\lambda)
    """
    w = np.zeros((X.feature_vector_len()))

    def epsilon_greedy_policy(s,w,epsilon=.0):
        nA = env.nA 
        #print(len(w))
        Q = [np.dot(w, X(s,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    for i in range(0, num_episode):
        state = env.reset()
        action = epsilon_greedy_policy(state, w, epsilon)

        x = X(state, action) #features

        while True:
            new_state, reward, done, goal = env.step(action)

            new_action = epsilon_greedy_policy(new_state, w, epsilon)

            new_x = X(new_state, new_action)

            val = np.dot(w, x)

            new_val = np.dot(w, new_x)

            if done and goal:
                delta = reward + goal_reward - val
            elif done:
                delta = reward - eta - val
            else:
                delta = reward + gamma*new_val - val

            w = w + alpha*delta*x

            x = new_x
            action = new_action

            if done:
                break

    pi_star = GreedyPolicy(env.nA, w, X)

    return w, pi_star


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
            

    