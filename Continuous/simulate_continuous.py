import numpy as np

from env import Env 
from policy import Policy

from features import StateFeatureVector

def Simulate_Semigradient_TD(env: Env,
	policy: Policy,
	num_episode: int,
	X: StateFeatureVector,
	gamma:float, # discount factor
	alpha:float) -> float: # step size

	w = np.zeros((X.feature_vector_len()))

	for i in range(0, num_episode):
		state = env.reset()
		done = False
		action = policy.action(state)

		x = X(state, done) #features

		old_val = 0.

		while True:
			new_state, reward, done, goal = env.step(action)

			action = policy.action(new_state)

			new_x = X(new_state, done)

			val = np.dot(w, x)

			new_val = np.dot(w, new_x)

			if done and goal:
				delta = gamma*new_val - val
			elif done:
				delta = 1 + gamma*new_val - val
			else:
				delta = gamma*new_val - val

			w = w + alpha*delta*x

			old_val = new_val

			x = new_x

			if done:
				break

	v_start_state = np.dot(X(env.reset(), False), w)
	return v_start_state



