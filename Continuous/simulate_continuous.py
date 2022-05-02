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

		while True:
			# print("I am here")
			action = policy.action(state)
			x = X(state) #features
			
			new_state, reward, done, goal = env.step(action)
			
			new_x = X(new_state)

			val = np.dot(w, x)

			new_val = np.dot(w, new_x)

			if done and goal:
				delta =  - val
			elif done:
				delta = 1 - val
			else:
				delta = gamma*new_val - val

			w = w + alpha*delta*x

			x = new_x
			state = new_state

			if done:
				break


	v_start_state = np.dot(X(env.reset()), w)
	return v_start_state



