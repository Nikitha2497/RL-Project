import sys
import os
sys.path.append('../')

import numpy as np
import math

from env_continuous import ContinuousEnv
from region import Rectangle

from features_poly import StateActionFeatureVectorWithPoly
from features_poly import StateFeatureVectorWithPoly 

from Sarsa import Sarsa
from simulate_continuous import Simulate_Semigradient_TD

#region env
#There is no state cost here
gamma = 1
alpha = 0.5 
epsilon = 0.1 
noise_std = math.sqrt(0) #Noise standard deviation,
noise_mean = 0 #Noise mean
boundary =  Rectangle(0,0,0.7,0.7)
not_safe_regions = [] #List of non safe Rectangles
not_safe_regions.append(Rectangle(0.1,0.1,0.3,0.5))
goal = Rectangle(0.1,0,0.2,0.1)
start_state = tuple((0.1,0.6)) #Initial state

beta1 =  0.01 #step size in horizontal direction
beta2 = 0.01 #step size in vertical direction
lambda1 = 0.5 #control cost
goal_reward = 5; #terminal reward
eta = 10
num_episode = 1000
########################################################

env = ContinuousEnv(lambda1,
    noise_std,
    noise_mean,
    beta1,
    beta2,
    boundary,
    not_safe_regions,
    goal,
    start_state)

runs = 1

failure_prob_with_eta = {}

num_episode_simulated = 1000

nA = 4
X_state_action = StateActionFeatureVectorWithPoly(4)
X_state = StateFeatureVectorWithPoly()


#create a results folder if one doesn't exist to store the plot figures
if not os.path.exists('results'):
    os.makedirs('results')

for run in range(0,runs):
    print("############run", run, "#################")
    (w_star, pi_star) = Sarsa(env,  gamma,
        alpha,
        X_state_action,
        num_episode,
        eta,
        goal_reward,
        epsilon)

    print(w_star)
#     print(pi_star)

    # eta = eta+5

#     failure_prob = Simulate_Semigradient_TD(env, 
#         pi_star,
#         num_episode_simulated,
#         X_state,
#         gamma,
#         alpha)

#     failure_prob_with_eta[eta] = failure_prob

#     print("failure_prob", failure_prob)



