import numpy as np
from env import Env

from typing import Tuple

from region import Rectangle

#This is continuous space and discrete action space env

#There is no state cost in this

class ContinuousEnv(Env):
	def __init__(self, lambda1,
		noise_std:float, #Noise standard deviation,
		noise_mean: float, #Noise mean
		beta1:float, #step size in horizontal direction
		beta2:float, #step size in vertical direction
		boundary: Rectangle,
		not_safe_regions, #List of Rectangles
		goal: Rectangle,
		start_state: Tuple[int, int], #Initial state
		):

		self.boundary = boundary
		self.not_safe_regions = not_safe_regions
		self.goal = goal

		self.beta1 = beta1
		self.beta2 = beta2

		self.lambda1 = lambda1

		self.noise_std = noise_std
		self.noise_mean = noise_mean

		self.start_state = start_state

		self._state = self.start_state

		#Number of actions are always 4 for the discrete action space
		self._nA = 4


	@property
	def nA(self) -> int:
		""" # possible actions """
		return self._nA

	def reset(self):
		self._state = self.start_state
		return self._state

	def step(self, action):
		assert action in list(range(self._nA)), "Invalid Action"

		new_x = self._state[0] + np.random.normal(self.noise_mean, self.noise_std)
		new_y = self._state[1] + np.random.normal(self.noise_mean, self.noise_std)

		#Actions
		#0 - North. 1 - West, 2 - South, 3 - East 
		
		if action == 0:
			new_y += self.beta2

		elif action == 1:
			new_x -= self.beta1

		elif action == 2:
			new_y -= self.beta2

		else:
			new_x += self.beta1

		self._state = tuple((new_x, new_y))

		reward = -1*self.lambda1

		#Not inside the boundary
		if not self.boundary.is_in(new_x, new_y):
			return self._state, reward,  True, False 

		#If the point is inside the red regions
		for region in self.not_safe_regions:
			if region.is_in(new_x, new_y):
				return self._state, reward, True, False

		#If the point in the goal region
		if self.goal.is_in(new_x, new_y):
			return self._state, reward, True, True

		return self._state, reward, False, False

		

		