import numpy as np

from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning

from simulate import Simulate

import matplotlib.pylab as plt


lambda1 = 0 #control cost
lambda2 = 2 #state cost

primary_prob = 0.9
secondary_prob = 0.05

gamma = 1
alpha = 0.1 #not using this, alpha is set to 1/t
num_episode = 100000
#eta = 100 - second path
eta = 60
epsilon = 0.01 #not using this, epsilon is set to 1/(epsilon_factor*t)
epsilon_factor = 1

max_num_steps = 100 #This is to check if there is a loop or not

runs = 10

max_simulated_runs = 100

failure_prob_with_eta = {}

for run in range(0, runs):
	print("############run", run, "#################")
	env = DiscreteEnv(lambda1, lambda2, primary_prob, secondary_prob)

	initQ = np.zeros((env.nS, env.nA))

	

	(Q_star, pi_star) = QLearning(env, 
	    gamma,
	    alpha,
	    num_episode,
	    eta,
	    initQ,
	    epsilon,
	    epsilon_factor)
	
	# eta = eta + 100

	# print(Q_star)
	print(Q_star[9])
	pi_star.print_all()

	failed_runs = 0
	for simulate_run in range(0, max_simulated_runs):
		result = Simulate(env, pi_star, max_num_steps)

		if (result == 0) :
			failed_runs += 1
		elif(result == -1) :
			print("Looks like there is a loop in the policy")


	failure_prob = failed_runs/max_simulated_runs

	failure_prob_with_eta[eta] = failure_prob

	print("failure_prob", failure_prob, failed_runs)


# #Plotting eta with failure probability
# lists = sorted(failure_prob_with_eta.items())

# x, y = zip(*lists)

# plt.plot(x, y)
# plt.show()





