import sys
sys.path.append('../')

import random

import numpy as np
import os
import math
from region import Rectangle
from env_continuous import ContinuousEnv
from simulate_continuous import Simulate_Semigradient_TD

import matplotlib.pylab as plt

from plot import compare_plot_CI
from plot import plot_CI
from plot import compare_plot_CI_seaborn

from policy import Policy
from restoredgreedypolicy import RestoredGreedyPolicy

from features_tilecoding import StateFeatureVectorWithTile
from features_tilecoding import StateActionFeatureVectorWithTile
from restoredgreedypolicy import RestoredGreedyPolicy


#Use tex for labelling the plots
plt.rcParams['text.usetex'] = True

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

#endregion env

#region tile coding features
state_low  = np.array([0, 0])
state_high = np.array([0.7, 0.7])
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
#end region Tile coding

runs = 10
num_episode_simulated = 100000
nA = 4

pi_star = RestoredGreedyPolicy(nA, X_state_action)
eta1 = 30
pi_star.restore_fromfile('data/pi_star_' + str(eta1) + ".txt")

failure_prob_runs1 = np.zeros((runs, num_episode_simulated))
for run in range(0, runs):
    random.seed(run)
    failure_prob, v_star_start_TD = Simulate_Semigradient_TD(env, 
        pi_star,
        num_episode_simulated,
        X_state,
        gamma,
        alpha)
    failure_prob_runs1[run] = v_star_start_TD



pi_star = RestoredGreedyPolicy(nA, X_state_action)
eta2 = 100
pi_star.restore_fromfile('data/pi_star_' + str(eta2) + ".txt")

failure_prob_runs2 = np.zeros((runs, num_episode_simulated))
for run in range(0, runs):
    random.seed(run)
    failure_prob, v_star_start_TD = Simulate_Semigradient_TD(env, 
        pi_star,
        num_episode_simulated,
        X_state,
        gamma,
        alpha)
    failure_prob_runs2[run] = v_star_start_TD


compare_plot_CI(failure_prob_runs1, r'$\eta$ = ' + str(eta1) ,
    failure_prob_runs2, r'$\eta$ = ' + str(eta2) , 
    r'\textbf{Epsiodes}', r'\textbf{$P_{fail}$}', 
    'results/Pfail_' + str(eta1) + '_'+ str(eta2) + '.png', True)


compare_plot_CI_seaborn(failure_prob_runs1, r'$\eta$ = ' + str(eta1) ,
    failure_prob_runs2, r'$\eta$ = ' + str(eta2) , 
    r'\textbf{Epsiodes}', r'\textbf{$P_{fail}$}', 
    'results/Pfail_lineplot_95confidence_' + str(eta1) + '_'+ str(eta2) + '.png', True)



    



