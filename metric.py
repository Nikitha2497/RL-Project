import numpy as np

#This class stores the metrics that we want to plot

class Metric():
	def __init__(self, num_episode:int):
		self.v_star_start = np.zeros(num_episode)
		self.q_star_start = {}
		for i in range(0, 4):
			self.q_star_start[i] = np.zeros(num_episode)

		self.a_star_start = np.zeros(num_episode)
		self.num_steps = np.zeros(num_episode)


	def set_v_star_start(self, index, value):
		self.v_star_start[index]  = value

	def get_v_star_start(self) -> np.array:
		return self.v_star_start


	def set_q_star_start(self, action:int, index, value):
		self.q_star_start[action][index] = value

	def get_q_star_start(self, action:int) -> np.array:
		return self.q_star_start[action]


	def set_a_star_start(self, index, value):
		self.a_star_start[index] = value

	def get_a_star_start(self) -> np.array:
		return self.a_star_start 

	def set_num_steps(self, index, value):
		self.num_steps[index] = index

	def get_num_steps(self) -> np.array:
		return self.num_steps










