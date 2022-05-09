import sys
sys.path.append('../')

import random

import numpy as np
import os
from env_discrete import DiscreteEnv
from simulate import Simulate_MC

from simulate import Simulate_TD

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import colors

from plot import compare_plot_CI
from plot import plot_CI
from plot import compare_plot_CI_seaborn

from policy import Policy

from restoredgreedypolicy import RestoredGreedyPolicy


#Use tex for labelling the plots
plt.rcParams['text.usetex'] = True

#region env
lambda1 = 1#control cost
lambda2 = 0 #state cost


primary_prob = 0.9

start_state = 10

state_matrix = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
						  [-1, 0, 2, 0, 0, 0, 0, -1],
						  [-1, 0, -1, -1, 0, 0, 0, -1],
						  [-1, 0, -1, -1, 0, 0, 0, -1],
						  [-1, 0, -1, -1, 0, 0, 0, -1],
						  [-1, 0, -1, -1, 0, 0, 0, -1],
						  [-1, 0, 1, 0, 0, 0, 0, -1],
						  [-1, -1, -1, -1, -1, -1, -1, -1]]) 

env = DiscreteEnv(lambda1, lambda2, primary_prob, state_matrix)

#endregion


states = state_matrix.shape[0]*state_matrix.shape[1]

pi_star = RestoredGreedyPolicy(states)
eta1 = 15
pi_star.restore_fromfile('data/pi_star_' + str(eta1) + ".txt")


start_state = 10
gamma = 1
alpha = 0.5 

max_num_steps = 100

runs = 10

num_episode_simulated = 100000

failure_prob_runs1 = np.zeros((runs, num_episode_simulated))
#Put the random seed
for run in range(0,runs):
	random.seed(run)
	failure_prob, failure_prob_array = Simulate_TD(env, pi_star, num_episode_simulated, gamma, alpha, start_state, max_num_steps, run)
	failure_prob_runs1[run] = failure_prob_array


eta2 = 50
pi_star = RestoredGreedyPolicy(states)
pi_star.restore_fromfile('data/pi_star_' + str(eta2) + ".txt")


failure_prob_runs2 = np.zeros((runs, num_episode_simulated))

for run in range(0,runs):
	random.seed(run)
	failure_prob, failure_prob_array = Simulate_TD(env, pi_star, num_episode_simulated, gamma, alpha, start_state, max_num_steps, run)
	failure_prob_runs2[run] = failure_prob_array


compare_plot_CI(failure_prob_runs1, r'$\eta$ = ' + str(eta1) ,
	failure_prob_runs2, r'$\eta$ = ' + str(eta2) , 
	r'\textbf{Epsiodes}', r'\textbf{$P_{fail}$}', 
	'results/Pfail_' + str(eta1) + '_'+ str(eta2) + '.png', True)

compare_plot_CI_seaborn(failure_prob_runs1, r'$\eta$ = ' + str(eta1) ,
	failure_prob_runs2, r'$\eta$ = ' + str(eta2) , 
	r'\textbf{Epsiodes}', r'\textbf{$P_{fail}$}', 
	'results/Pfail_lineplot_95confidence_' + str(eta1) + '_'+ str(eta2) + '.png', True)







