import sys
sys.path.append('../')

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

#region env
lambda1 = 1 #control cost
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
eta = 10

gamma = 1
alpha = 0.5 #not using this, alpha is set to 1/itr
num_episode = 100000
epsilon = 0.1 #not using this, epsilon is set to 1/itr

max_num_steps = 100 #This is to check if there is a loop or not
runs = 1
#max_simulated_runs = 100
######################################################################################

failure_prob_with_eta = {}

num_episode_simulated = 10000


#create a results folder if one doesn't exist to store the plot figures
if not os.path.exists('results'):
	os.makedirs('results')


for run in range(0, runs):
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

# 	eta = eta + 5
	
# 	V = np.zeros(Q_star.shape[0])

# 	for state in range(0, Q_star.shape[0]):
# 		V[state] = max(Q_star[state])

	# print(Q_star[env.reset()])
	# pi_star.print_all()
# 	plt.figure(1 + run)
# 	plt.plot(metric.get_v_star_start())
# 	plt.ylabel('V star start')
# 	plt.savefig('results/run_' + str(run) +  '_v_star.png')
# 	plt.clf()
# 	plt.figure(2 + run)
# 	plt.plot(metric.get_a_star_start())
# 	plt.ylabel('a star start')
# 	plt.savefig('results/run_' + str(run) +  '_a_star.png')
# 	plt.clf()
	plt.figure(3 + run)
	plt.plot(metric.get_q_star_start(1), label='W')
	plt.plot(metric.get_q_star_start(3), label='E')
	plt.legend(loc="upper right")
	plt.ylabel('Q (W, E)')
	plt.show()    
# 	plt.savefig('results/run_' + str(run) +  '_q_star.png')
	


# 	#Plot the surface plot for V
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	x =  np.arange(0, state_matrix.shape[0] - 1, 1)
# 	y =  np.arange(0, state_matrix.shape[1] - 1, 1)
# 	X, Y = np.meshgrid(x, y)
# 	zs = np.array(V[np.ravel(X)*state_matrix.shape[1] + np.ravel(Y)])
# 	Z = zs.reshape(X.shape)

# 	ax.plot_surface(X, Y, Z)

# 	ax.set_xlabel('X')
# 	ax.set_ylabel('Y')
# 	ax.set_zlabel('V_star')

# 	plt.savefig('results/run_' + str(run) +  '_v.png')
# 	plt.clf()
# 	plt.close()
	
	#TODO - Plot the policy graph
	
	# fig, ax = plt.subplots()
	# ax.imshow(data, cmap=cmap, norm=norm)

	# # draw gridlines
	# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	# ax.set_xticks(np.arange(-.5, 10, 1));
	# ax.set_yticks(np.arange(-.5, 10, 1));

# 	plt.figure(4)
# 	plt.plot(num_steps)
# 	plt.ylabel('num steps')
	# plt.show()


# 	failed_runs = 0
# 	for simulate_run in range(0, max_simulated_runs):
# 		result = Simulate_MC(env, pi_star, max_num_steps)

# 		if (result == 0) :
# 			failed_runs += 1
# 		elif(result == -1) :
# 			print("Looks like there is a loop in the policy")


# 	failure_prob = failed_runs/max_simulated_runs

# 	failure_prob = Simulate_TD(env, pi_star, num_episode_simulated, gamma, alpha, start_state, max_num_steps, run)

# 	failure_prob_with_eta[eta] = failure_prob

# 	print("failure_prob", failure_prob)


# # #Plotting eta with failure probability
# lists = sorted(failure_prob_with_eta.items())

# x, y = zip(*lists)

# plt.plot(x, y)
# plt.savefig('results/failure_prob.png')
# # plt.show()




