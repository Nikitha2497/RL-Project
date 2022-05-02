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
		# print(i)
		while True:
			# print("I am here")
			action = policy.action(state)
			# print("current state ", state, action)
			x = X(state) #features
			
			new_state, reward, done, goal = env.step(action)
			# print("new state", new_state, reward, done, goal)
			
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

	print("inside simulate")
	print(w)
	print(X(env.reset()))
	v_start_state = np.dot(w, X(env.reset()))

	return v_start_state



