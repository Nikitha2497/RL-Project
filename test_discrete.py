import numpy as np
from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning
from simulate import Simulate_MC

from simulate import Simulate_TD

import matplotlib.pylab as plt

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
eta = 50

gamma = 1
alpha = 0.5 #not using this, alpha is set to 1/itr
num_episode = 100000
epsilon = 0.1 #not using this, epsilon is set to 1/itr

#max_num_steps = 100 #This is to check if there is a loop or not
runs = 10
#max_simulated_runs = 100
######################################################################################

failure_prob_with_eta = {}

num_episode_simulated = 1000


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

	eta = eta + 10
	
	# print(Q_star[env.reset()])
	# pi_star.print_all()
	# plt.figure(1)
	# plt.plot(metric.get_v_star_start())
	# plt.ylabel('V star start')
	# plt.figure(2)
	# plt.plot(metric.get_a_star_start())
	# plt.ylabel('a star start')
	# plt.figure(3)
	# plt.plot(metric.get_q_star_start(1), label='W')
	# plt.plot(metric.get_q_star_start(3), label='E')
	# plt.legend(loc="upper right")
	# plt.ylabel('Q (W, E)')
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

	failure_prob = Simulate_TD(env, pi_star, num_episode_simulated, gamma, alpha, start_state)

	failure_prob_with_eta[eta] = failure_prob

	print("failure_prob", failure_prob)


# #Plotting eta with failure probability
lists = sorted(failure_prob_with_eta.items())

x, y = zip(*lists)

plt.plot(x, y)
plt.show()




