import sys
sys.path.append('../')
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib.patches as patches


import os
from env_continuous import ContinuousEnv

from restoredgreedypolicy import RestoredGreedyPolicy
from region import Rectangle
from typing import Tuple

from features_tilecoding import StateFeatureVectorWithTile
from features_tilecoding import StateActionFeatureVectorWithTile


#Plots the continuous domain, no noise trajectories and sample trajectories for the continuous state space
plt.rcParams['text.usetex'] = True
fontsize = 16

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


ticksize = 0.1

max_allowed_steps = 100


#Plotting the continuous domain
def setup_domain():
	fig, ax = plt.subplots()

	ax.set_xticks(np.arange(boundary.get_x1() - ticksize, boundary.get_x2() + ticksize, ticksize))
	ax.set_yticks(np.arange(boundary.get_y1() - ticksize, boundary.get_y2() + ticksize, ticksize))

	plt.plot(start_state[0], start_state[1], 'o')
	plt.text(start_state[0], start_state[1] + 0.02, 'I', fontsize = 22)
	# ax.annotate('(0.25, 0.65)', 
	#             xy=(start_state[0], start_state[1] - 0.03), 
	#             xytext = (start_state[0], start_state[1] - 0.03),
	#             ha='center', 
	#             va='center')

	plt.ylim(boundary.get_y1() - ticksize, boundary.get_y2() + ticksize)
	plt.xlim(boundary.get_x1() - ticksize , boundary.get_x2() + ticksize)

	boundary_width = boundary.width() + 2*ticksize
	boundary_length = boundary.length() + 2*ticksize
	#left side
	rect = patches.Rectangle((boundary.get_x1() - ticksize, boundary.get_y1() - ticksize), ticksize, boundary_width, linewidth=1, color='#A2142F')
	ax.add_patch(rect)
	#top
	rect = patches.Rectangle((boundary.get_x1() - ticksize, boundary.get_y2()), boundary_length, ticksize, linewidth=1, color='#A2142F')
	ax.add_patch(rect)
	#right side
	rect = patches.Rectangle((boundary.get_x2() , boundary.get_y1()  - ticksize), ticksize, boundary_width, linewidth=1, color='#A2142F')
	ax.add_patch(rect)
	#bottom
	rect = patches.Rectangle((boundary.get_x1() - ticksize, boundary.get_y1() - ticksize), boundary_length, ticksize, linewidth=1, color='#A2142F')
	ax.add_patch(rect)

	#Red regions
	for region in not_safe_regions:
		rect = patches.Rectangle((region.get_x1() , region.get_y1()), region.length(), region.width(), linewidth=1, color='#A2142F')
		ax.add_patch(rect)

	#Goal region
	rect = patches.Rectangle((goal.get_x1() , goal.get_y1()), goal.length(), goal.width(), linewidth=1, color='g')
	ax.add_patch(rect)

	goal_center = goal.center()
	plt.text(goal_center[0] - 0.01, goal_center[1] - 0.02, 'G', fontsize = 22)

	return fig, ax

def save_figure(filename):
	plt.savefig(filename + '.png')
	plt.savefig(filename + '.svg')

#setup domain
setup_domain()
plt.savefig('results/domain.png')

plt.grid(b=True,which='both')
plt.savefig('results/domain_withgrid.png')
plt.clf()

#Plots a single trajectory
def plot_line(pi_star, allow_noise=False) -> Tuple[list, list]:
	x = []
	y = []

	state = env.reset()

	x.append(state[0])
	y.append(state[1])
	env.reset()
	steps = 0
	while True:
		steps += 1
		action = pi_star.action(state)

		new_state, reward, done, goal = env.step(action,allow_noise)
		x.append(new_state[0])
		y.append(new_state[1])

		if done and goal:
			return x,y,False

		if done:
			return x,y,True

		if steps == max_allowed_steps:
			break

		state = new_state

	return x,y,False


#Plot the no-noise trajectory
setup_domain()
pi_star = RestoredGreedyPolicy(nA, X_state_action)
pi_star.restore_fromfile('data/pi_star_' + str(eta1) + ".txt")
(x,y,colliding) = plot_line(pi_star)
plt.plot(x,y, color = 'blue', label = r'$\eta$ = '+ str(eta1))
final_i = x[len(x) - 1]
final_j = y[len(y) - 1]
plt.arrow(final_i - 0.07, final_j, 0.01, 0,  head_width=0.04, head_length=0.02, linewidth=1, color='blue', length_includes_head=True)

pi_star = RestoredGreedyPolicy(nA, X_state_action)
pi_star.restore_fromfile('data/pi_star_' + str(eta2) + ".txt")
(x,y,colliding) = plot_line(pi_star)
plt.plot(x,y, color = 'orange', label = r'$\eta$ = '+ str(eta2))
final_i = x[len(x) - 1]
final_j = y[len(y) - 1]
plt.arrow(final_i+0.07, final_j, -0.01, 0,  head_width=0.04, head_length=0.02, linewidth=1, color='orange', length_includes_head=True)
plt.legend(loc = "upper right", fontsize = fontsize)
save_figure('results/no_traj_' + str(eta1) + '_' + str(eta2))
plt.clf()


#Plot with noise trajectories for num_simulations
num_simulations = 100
setup_domain()

pi_star1= RestoredGreedyPolicy(nA, X_state_action)
pi_star1.restore_fromfile('data/pi_star_' + str(eta1) + ".txt")
(x,y,colliding) = plot_line(pi_star1, True)
plt.plot(x,y, color = 'blue', label = r'$\eta$ = '+ str(eta1))


pi_star2 = RestoredGreedyPolicy(nA, X_state_action)
pi_star2.restore_fromfile('data/pi_star_' + str(eta2) + ".txt")
(x,y,colliding) = plot_line(pi_star2,True)
plt.plot(x,y, color = 'orange', label = r'$\eta$ = '+ str(eta2))

alpha = 0.5

for simulation in range(1, num_simulations):
	(x,y,colliding) = plot_line(pi_star1, True)
	plt.plot(x,y, color = 'blue', alpha = alpha)
	(x,y,colliding) = plot_line(pi_star2, True)
	plt.plot(x,y, color = 'orange', alpha = alpha)

plt.legend(loc = "upper right", fontsize = fontsize)
save_figure('results/with_traj_' + str(eta1) + '_' + str(eta2))
plt.clf()


#Draw with noise trajectory with different colors for colliding one and non-colliding
setup_domain()
pi_star1= RestoredGreedyPolicy(nA, X_state_action)
pi_star1.restore_fromfile('data/pi_star_' + str(eta1) + ".txt")
for simulation in range(0, num_simulations):
	(x,y,colliding) = plot_line(pi_star1, True)
	if colliding:
		plt.plot(x,y, color = 'magenta')
	else:
		plt.plot(x,y,color = 'blue')
plt.savefig('results/with_traj_' + str(eta1) + '.png')
plt.clf()


setup_domain()

pi_star1= RestoredGreedyPolicy(nA, X_state_action)
pi_star1.restore_fromfile('data/pi_star_' + str(eta2) + ".txt")
for simulation in range(0, num_simulations):
	(x,y,colliding) = plot_line(pi_star1, True)
	if colliding:
		plt.plot(x,y, color = 'magenta')
	else:
		plt.plot(x,y,color = 'blue')
plt.savefig('results/with_traj_' + str(eta2) + '.png')
plt.clf()










