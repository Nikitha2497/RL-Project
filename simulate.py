import numpy as np

from env import Env 
from policy import Policy

def Simulate(env: Env, policy: Policy, max_num_steps:int) -> int:
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


