import sys
sys.path.append('../')

import random

import numpy as np
import os
from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning
from simulate import Simulate_MC

from simulate import Simulate_TD

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import colors

from metric import Metric

# Use tex for labelling the plots
plt.rcParams['text.usetex'] = True

from plot import compare_plot_CI
from plot import plot_CI
from plot import compare_plot_CI_seaborn

#Run file for the control algorithm

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

goal_reward = 5; #terminal reward
eta1 = 15 #Takes West
eta2 = 50 #Takes East

gamma = 1
alpha = 0.5 #not using this, alpha is set to 1/itr
num_episode = 100000
epsilon = 0.1 #not using this, epsilon is set to 1/itr

runs = 10
######################################################################################


#create a results folder if one doesn't exist to store the plot figures
if not os.path.exists('results'):
	os.makedirs('results')

if not os.path.exists('data'):
	os.makedirs('data')


def run_control(eta):
	#This is used for final plotting
	final_q_star_W_episodes = np.zeros((runs, num_episode))
	final_q_star_E_episodes = np.zeros((runs, num_episode))


	for run in range(0, runs):
		random.seed(run)
		print("############run", run, "#################")
		initQ = np.zeros((env.nS, env.nA))
	    
		(Q_star, pi_star, metric) = QLearning(env, 
		    gamma,
		    alpha,
		    num_episode,
		    eta,
		    goal_reward,
		    initQ,
		    epsilon,
		    False)

		pi_star.print_all()

		if run == 0:
			pi_star.save_tofile("data/pi_star_" + str(eta) + ".txt")


		final_q_star_W_episodes[run] = metric.get_q_star_start(1)
		final_q_star_E_episodes[run] = metric.get_q_star_start(3)

	ci = "sd"

	compare_plot_CI_seaborn(final_q_star_W_episodes, 'Q(I, W)' ,
		final_q_star_E_episodes, 'Q(I, E)', 
		r'\textbf{Epsiodes}', r'\textbf{Q (I , $\bullet$)}', 
		'results/q_line_' + str(ci) + '_ci' +str(eta) , ci)

	compare_plot_CI(final_q_star_W_episodes, 'Q(I, W)' ,
		final_q_star_E_episodes, 'Q(I, E)', 
		r'\textbf{Epsiodes}', r'\textbf{Q (I , $\bullet$)}', 
		'results/q_' + str(eta) )

	

#Run discrete env control for eta1 and eta2
run_control(eta1)
run_control(eta2)









