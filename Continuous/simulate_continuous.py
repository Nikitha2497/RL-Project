import numpy as np
from env import Env 
from policy import Policy

from features import StateFeatureVector

from typing import Tuple

def Simulate_Semigradient_TD(env: Env,
	policy: Policy,
	num_episode: int,
	X: StateFeatureVector,
	gamma:float, # discount factor
	alpha:float) -> Tuple[float, np.array]: # step size

	w = np.zeros((X.feature_vector_len()))
    
	v_star_start = np.zeros((num_episode))

	itr = 1; #to decay epsilon and alpha
	for i in range(0, num_episode):
		state = env.reset()
		# print(i)
		if (i%(num_episode/100)==0):
			epsilon = 1./(itr)
			alpha = 1./(itr)
			itr += 1
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

		v_star_start[i] = np.dot(w, X(env.reset()))


	print("inside simulate")
	print(w)
	print(X(env.reset()))
	v_start_state = np.dot(w, X(env.reset()))

	return v_start_state, v_star_start



