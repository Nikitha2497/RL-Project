import numpy as np

from env import Env 
from policy import Policy

import matplotlib.pylab as plt

from typing import Tuple

#Implementation of modified TD(0) and MC for risk estimation of the policy(probability failure prediction)
def Simulate_MC(env: Env, policy: Policy, max_num_steps:int) -> int:
	steps = 0
	state = env.reset()
	while True:
		steps += 1
		action = policy.action(state)

		new_state, cost, done, goal = env.step(action)

		if done and goal:
			return 1

		if done:
			return 0

		if steps == max_num_steps:
			return -1	

		state = new_state


def Simulate_TD(env: Env,
				 policy: Policy,
				 num_episode: int,
				 gamma: int,
				 alpha: int,
				 start_state:int,
				 max_num_steps: int,
				 run:int) -> Tuple[float, np.array]:
	V = np.zeros(env._nS)

	v_star = np.zeros(num_episode)

	itr = 1; #to decay epsilon and alpha


	for episode in range(0, num_episode):
		state = env.reset()
		steps = 0

		if (episode%(num_episode/100)==0):
			epsilon = 1./(itr)
			alpha = 1./(itr)
			itr += 1

		while True:
			steps += 1
			action = policy.action(state)

			new_state, cost, done, goal = env.step(action)

			if done and goal:
				V[state] = V[state] - alpha*V[state]
				break

			if done:
				V[state] = V[state] + alpha*(1 - V[state])
				break

			V[state] = V[state] + alpha*(gamma*V[new_state] - V[state])

			if steps ==  max_num_steps:
				V[state] = V[state] + alpha*(1 - V[state])
				print("Looks like this policy is looping")
				break

			state = new_state

		v_star[episode] = V[start_state]

	return V[start_state], v_star





