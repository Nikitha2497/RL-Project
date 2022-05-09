import sys
import os
sys.path.append('../')

import numpy as np
import math

from env_continuous import ContinuousEnv
from region import Rectangle

from features_poly import StateActionFeatureVectorWithPoly
from features_poly import StateFeatureVectorWithPoly 


from features_tilecoding import StateActionFeatureVectorWithTile
from features_tilecoding import StateFeatureVectorWithTile

from Sarsa import Sarsa
from simulate_continuous import Simulate_Semigradient_TD
from metric import Metric
import matplotlib.pylab as plt

#region env
#There is no state cost here
gamma = 1
alpha = 0.5 
epsilon = 0.1
noise_std = math.sqrt(0.001) #Noise standard deviation,
noise_mean = 0 #Noise mean
boundary =  Rectangle(0.1,0.1,0.7,0.7, True) #The outer boundary
not_safe_regions = [] #List of non safe Rectangles
not_safe_regions.append(Rectangle(0.2,0.2,0.4,0.6))
goal = Rectangle(0.2, 0.1, 0.3, 0.2)
start_state = tuple((0.25,0.65)) #Initial state

beta1 =  0.1 #step size in horizontal direction
beta2 = 0.1 #step size in vertical direction
lambda1 = 1 #control cost
goal_reward = 10; #terminal reward
eta = 100 #W-40 E-100
num_episode = 100000
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

# failure_prob_with_eta = {}

num_episode_simulated = 1000

##polynomial features
# nA = 4
# X_state_action = StateActionFeatureVectorWithPoly(4)
# X_state = StateFeatureVectorWithPoly()

#region tile coding features
state_low  = np.array([0, 0])
state_high = np.array([1, 1])
nA = 4
num_tilings = 1
tile_width = np.array([0.1, 0.1])

X_state_action = StateActionFeatureVectorWithTile(state_low,
                 state_high,
                 nA,
                 num_tilings,
                 tile_width)
X_state = StateFeatureVectorWithTile(state_low,
                 state_high,
                 num_tilings,
                 tile_width)

#create a results folder if one doesn't exist to store the plot figures
# if not os.path.exists('results'):
#     os.makedirs('results')

for run in range(0,runs):
    print("############run", run, "#################")
    (w_star, pi_star, metric) = Sarsa(env,  gamma,
        alpha,
        X_state_action,
        num_episode,
        eta,
        goal_reward,
        epsilon)

#     print(w_star)
#     print(pi_star)

    plt.figure(1)
    plt.plot(metric.get_v_star_start())
    plt.ylabel('V star start')    
#     plt.figure(2 + run)
#     plt.plot(metric.get_a_star_start())
#     plt.ylabel('a star start')
    plt.figure(2)
    #plt.plot(metric.get_q_star_start(0), label='N')
    plt.plot(metric.get_q_star_start(1), label='W')
    # plt.plot(metric.get_q_star_start(2), label='S')
    plt.plot(metric.get_q_star_start(3), label='E')
    plt.legend(loc="upper right")
    plt.ylabel('Q (W, E)')
    plt.show()
    
    # eta = eta+5

    failure_prob, v_star_start_TD = Simulate_Semigradient_TD(env, 
        pi_star,
        num_episode_simulated,
        X_state,
        gamma,
        alpha)
    
    plt.figure(3)
    plt.plot(v_star_start_TD)
    plt.ylabel('V star start TD')
    plt.show()

#     failure_prob_with_eta[eta] = failure_prob

#     print("failure_prob", failure_prob)



