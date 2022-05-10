import numpy as np
from env import Env 
from policy import Policy

from features import StateFeatureVector

from typing import Tuple

#Implementation of modified semi gradient TD for risk estimation of the policy(probability failure prediction)
def Simulate_Semigradient_TD(env: Env,
	policy: Policy,
	num_episode: int,
	X: StateFeatureVector,
	gamma:float, # discount factor
	alpha:float) -> Tuple[float, np.array]: # step size

	w = np.zeros((X.feature_vector_len()))
    
	v_star_start = np.zeros((num_episode))

	itr = 1; #to decay alpha
	for i in range(0, num_episode):
		state = env.reset()
		# print(i)
		if (i%(num_episode/100)==0):
			alpha = 1./(itr+1)
			itr += 1
			print(i/(num_episode/100))
		while True:
			action = policy.action(state)
			x = X(state) #features
			
			new_state, reward, done, goal = env.step(action)
			
			new_x = X(new_state)

			val = np.dot(w, x)

			new_val = np.dot(w, new_x)

			#goal state
			if done and goal:
				delta =  - val
			#unsafe state
			elif done:
				delta = 1 - val
			#safe state
			else:
				delta = gamma*new_val - val

			w = w + alpha*delta*x

			x = new_x
			state = new_state

			if done:
				break

		v_star_start[i] = np.dot(w, X(env.reset()))

	v_start_state = np.dot(w, X(env.reset()))

	return v_start_state, v_star_start



