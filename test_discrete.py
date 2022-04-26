import numpy as np
from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning
from simulate import Simulate
import matplotlib.pylab as plt

lambda1 = 1 #control cost
lambda2 = 0 #state cost
goal_reward = 5; #terminal reward
eta = 50
primary_prob = 0.9
secondary_prob = (1-primary_prob)/2

gamma = 1
alpha = 0.5 #not using this, alpha is set to 1/itr
num_episode = 100000
epsilon = 0.1 #not using this, epsilon is set to 1/itr

#max_num_steps = 100 #This is to check if there is a loop or not
runs = 1
#max_simulated_runs = 100
######################################################################################

#failure_prob_with_eta = {}

env = DiscreteEnv(lambda1, lambda2, primary_prob, secondary_prob)

for run in range(0, runs):
	print("############run", run, "#################")
	initQ = np.zeros((env.nS, env.nA))
    
	(Q_star, pi_star, V_star_start, Q_W_start, Q_E_start, a_star_start, num_steps) = QLearning(env, 
	    gamma,
	    alpha,
	    num_episode,
	    eta,
	    goal_reward,
	    initQ,
	    epsilon)
	
	print(Q_star[env.reset()])
	pi_star.print_all()
	plt.figure(1)
	plt.plot(V_star_start)
	plt.ylabel('V star start')
	plt.figure(2)
	plt.plot(a_star_start)
	plt.ylabel('a star start')
	plt.figure(3)
	plt.plot(Q_W_start, label='W')
	plt.plot(Q_E_start, label='E')
	plt.legend(loc="upper right")
	plt.ylabel('Q (W, E)')
# 	plt.figure(4)
# 	plt.plot(num_steps)
# 	plt.ylabel('num steps')
	plt.show()


# 	failed_runs = 0
# 	for simulate_run in range(0, max_simulated_runs):
# 		result = Simulate(env, pi_star, max_num_steps)

# 		if (result == 0) :
# 			failed_runs += 1
# 		elif(result == -1) :
# 			print("Looks like there is a loop in the policy")


# 	failure_prob = failed_runs/max_simulated_runs

# 	failure_prob_with_eta[eta] = failure_prob

# 	print("failure_prob", failure_prob, failed_runs)


# #Plotting eta with failure probability
# lists = sorted(failure_prob_with_eta.items())

# x, y = zip(*lists)

# plt.plot(x, y)
# plt.show()




