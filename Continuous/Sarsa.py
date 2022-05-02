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
    epsilon=.0
) -> Tuple[np.array, Policy]:
    """
    Implement True online Sarsa(\lambda)
    """
    w = np.zeros((X.feature_vector_len()))

    epsilon = 0.

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.nA 
        #print(len(w))
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    for i in range(0, num_episode):
        #print(i)
        state = env.reset()
        done = False
        action = epsilon_greedy_policy(state, done, w, epsilon)

        x = X.__call__(state, done, action) #features

        old_val = 0.

        while True:
            new_state, reward, done, goal = env.step(action)

            action = epsilon_greedy_policy(new_state, done, w, epsilon)

            new_x = X.__call__(new_state, done, action)

            val = np.dot(w, x)

            new_val = np.dot(w, new_x)

            if done and goal:
                delta = reward + goal_reward + gamma*new_val - val
            elif done:
                delta = reward - eta + gamma*new_val - val
            else:
                delta = reward + gamma*new_val - val

            w = w + alpha*delta*x

            old_val = new_val

            x = new_x

            if done:
                break

    pi_star = GreedyPolicy(env, w, X)

    return w, pi_star


class GreedyPolicy(Policy):

    def __init__(self, env: Env, w:np.array, X:StateActionFeatureVector):
        self.env = env
        self.nA = env.nA
        self.w = w
        self.X = X

    def action_prob(self,state,action:int):
        done = self.env(state)
        Q = [np.dot(self.w, self.X(state,done,a)) for a in range(self.nA)]
        if action == np.argmax(Q):
            return 1.0
        return 0.0;

    def action(self,state):
        done = self.env.is_terminal(state)
        Q = [np.dot(self.w, self.X(state,done,a)) for a in range(self.nA)]
        return np.argmax(Q)
            

    