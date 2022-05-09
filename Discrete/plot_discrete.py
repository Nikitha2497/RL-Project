import sys
sys.path.append('../')
import numpy as np
import matplotlib.pylab as plt

import os
from env_discrete import DiscreteEnv

from restoredgreedypolicy import RestoredGreedyPolicy

from typing import Tuple

plt.rcParams['text.usetex'] = True

def plot_line(n, start_state, env, pi_star, allow_noise=False) -> Tuple[list, list]:
	x = []
	y = []

	state = start_state

	x.append(int(state/n) + 0.5)
	y.append(int(state % n) + 0.7)
	env.reset()
	while True:
		action = pi_star.action(state)

		new_state, reward, done, goal = env.step(action,allow_noise)
		x.append(int(new_state/n) + 0.5)
		y.append(int(new_state % n) + 0.5)

		if done:
			break

		state = new_state

	return y,x

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

#Set the red regions on the grid world
m = state_matrix.shape[0]
n = state_matrix.shape[1]
states = m*n 

start_state = 10
start_state_i = 2
start_state_j = 1
final_state_i = 2
final_state_j = 6 

red_regions = np.zeros((m,n))
for i in range(0, m):
	for j in range(0, n):
		if state_matrix[i][j] == -1:
			red_regions[i][j] = 1 


#Set the grid world on the plot
fig, ax = plt.subplots()

ax.set_xticks(np.arange(0, n, 1))
ax.set_yticks(np.arange(0, m, 1))
plt.text(start_state_i + 0.5, start_state_j + 0.7, 'I', fontsize = 22)
plt.text(final_state_i + 0.5, final_state_j + 0.7, 'G', fontsize = 22)
ax.pcolor(red_regions,  linestyle= 'dashed',  cmap='OrRd')
plt.ylim(0,m)
plt.xlim(0,n)
ax.invert_yaxis()
plt.grid(b=True,which='both')
plt.savefig('results/grid.png')

#Plot the first eta line
eta1 = 15
pi_star = RestoredGreedyPolicy(states)
pi_star.restore_fromfile('data/pi_star_' + str(eta1) + '.txt')
(x,y) = plot_line(n, start_state, env, pi_star)
plt.plot(x,y, color = 'blue', label = r'$\eta$ = '+ str(eta1))

#Plot the eta2 line
eta2 = 50
pi_star = RestoredGreedyPolicy(states)
pi_star.restore_fromfile('data/pi_star_' + str(eta2) + '.txt')
(x,y) = plot_line(n, start_state, env, pi_star)
plt.plot(x,y, color = 'orange', label = r'$\eta$ = '+ str(eta2))
plt.legend(loc = "upper right")
plt.savefig('results/grid_' + str(eta1) + '_'+ str(eta2) + '.png')


#Set the grid world on the plot
fig, ax = plt.subplots()
ax.set_xticks(np.arange(0, n, 1))
ax.set_yticks(np.arange(0, m, 1))
plt.text(start_state_i + 0.5, start_state_j + 0.7, 'I', fontsize = 22)
plt.text(final_state_i + 0.5, final_state_j + 0.7, 'G', fontsize = 22)
ax.pcolor(red_regions,  linestyle= 'dashed',  cmap='OrRd')
plt.ylim(0,m)
plt.xlim(0,n)
ax.invert_yaxis()
plt.grid(b=True,which='both')

num_simulations = 100000

eta1 = 15
eta2 = 50
pi_star1 = RestoredGreedyPolicy(states)
pi_star1.restore_fromfile('data/pi_star_' + str(eta1) + '.txt')
pi_star2 = RestoredGreedyPolicy(states)
pi_star2.restore_fromfile('data/pi_star_' + str(eta2) + '.txt')
(x,y) = plot_line(n, start_state, env, pi_star1, True)
plt.plot(x,y, color = 'blue', label = r'$\eta$ = '+ str(eta1), alpha = 1)
(x,y) = plot_line(n, start_state, env, pi_star2, True)
plt.plot(x,y, color = 'orange', label = r'$\eta$ = '+ str(eta2), alpha = 1)
alpha = 1
for simulation in range(1, num_simulations):
	alpha = alpha*0.9
	(x,y) = plot_line(n, start_state, env, pi_star1, True)
	plt.plot(x,y, color = 'blue', alpha = alpha)
	(x,y) = plot_line(n, start_state, env, pi_star2, True)
	plt.plot(x,y, color = 'orange', alpha = alpha)
plt.legend(loc="upper right")
plt.savefig('results/grid_sampletrajs_' + str(eta1) + '_'+ str(eta2) + '.png')
plt.close()















