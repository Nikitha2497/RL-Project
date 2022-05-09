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

import random


from plot import plot_CI
from plot import compare_plot_CI

from plot import compare_plot_CI_seaborn


#region env
#There is no state cost here
gamma = 1
alpha = 0.5 
epsilon = 0.1
noise_std = math.sqrt(0.0005) #Noise standard deviation,
noise_mean = 0 #Noise mean
boundary =  Rectangle(0.1,0.1,0.7,0.7, True) #The outer boundary
not_safe_regions = [] #List of non safe Rectangles
not_safe_regions.append(Rectangle(0.2,0.2,0.4,0.6))
goal = Rectangle(0.2, 0.1, 0.3, 0.2)
start_state = tuple((0.25,0.65)) #Initial state

beta1 =  0.1 #step size in horizontal direction
beta2 = 0.1 #step size in vertical direction
lambda1 = 1 #control cost
goal_reward = 5; #terminal reward
# eta = 70 #W-15 E-70
eta1 = 15
eta2 = 70
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
#endregion tile coding
############################
runs = 10

##polynomial features
# nA = 4
# X_state_action = StateActionFeatureVectorWithPoly(4)
# X_state = StateFeatureVectorWithPoly()


#create a results folder if one doesn't exist to store the plot figures
if not os.path.exists('results'):
	os.makedirs('results')

if not os.path.exists('data'):
	os.makedirs('data')


def run_control(eta):
	#This is used for final plotting
	final_q_star_W_episodes = np.zeros((runs, num_episode))
	final_q_star_E_episodes = np.zeros((runs, num_episode))


	for run in range(0,runs):
		random.seed(run)
		print("############run", run, "#################")
		(w_star, pi_star, metric) = Sarsa(env,  gamma,
			alpha,
			X_state_action,
			num_episode,
			eta,
			goal_reward,
			epsilon)


		if run == 0:
			print("Saved the pi_star weights to a file")
			pi_star.save_tofile('data/pi_star_' + str(eta) + '.txt')
			# print(w_star)

		final_q_star_W_episodes[run] = metric.get_q_star_start(1)
		final_q_star_E_episodes[run] = metric.get_q_star_start(3)

	ci = "sd"    

	compare_plot_CI_seaborn(final_q_star_W_episodes, 'Q(I, W)' ,
		final_q_star_E_episodes, 'Q(I, E)', 
		r'\textbf{Epsiodes}', r'\textbf{Q (I , $\bullet$)}', 
		'results/q_line_ ' + str(ci) + '_ci' +str(eta) , ci)
	

	# compare_plot_CI(final_q_star_W_episodes, 'Q(I, W)' ,
	#     final_q_star_E_episodes, 'Q(I, E)', 
	#     r'\textbf{Epsiodes}', r'\textbf{Q (I , $\bullet$)}', 
	#     'results/q_' + str(eta) )

	
#Run the sarsa control algorithm for both the eta1 and eta2
run_control(eta1)
run_control(eta2)

