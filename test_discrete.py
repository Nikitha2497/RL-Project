import numpy as np

from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning


lambda1 = 0 #control cost
lambda2 = 2 #state cost

primary_prob = 0.9
secondary_prob = 0.05

gamma = 1
alpha = 0.1 #not using this, alpha is set to 1/t
num_episode = 1000000
#eta = 100 - second path
eta = 60
epsilon = 0.01 #not using this, epsilon is set to 1/(epsilon_factor*t)
epsilon_factor = 10

runs = 10
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


	# print(Q_star)
	print(Q_star[9])
	pi_star.print_all()



