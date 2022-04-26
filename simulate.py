import numpy as np

from env import Env 
from policy import Policy

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
				 start_state:int) -> float:
	V = np.zeros(env._nS)

	for episode in range(0, num_episode):
		state = env.reset()
		steps = 0
		while True:
			action = policy.action(state)

			new_state, cost, done, goal = env.step(action)

			if done and goal:
				V[state] = V[state] - alpha*V[state]
				break

			if done:
				V[state] = V[state] + alpha*(1 - V[state])
				break

			V[state] = V[state] + alpha*(gamma*V[new_state] - V[state])


			state = new_state

	return V[start_state]





