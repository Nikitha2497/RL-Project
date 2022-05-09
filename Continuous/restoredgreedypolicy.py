import numpy as np

from policy import Policy
from features import StateActionFeatureVector

class RestoredGreedyPolicy(Policy):
	def __init__(self, nA: int, X:StateActionFeatureVector):
		self.nA = nA
		self.X = X

	def action_prob(self,state,action:int):
		Q = [np.dot(self.w, self.X(state,a)) for a in range(self.nA)]
		if action == np.argmax(Q):
			return 1.0
		return 0.0;

	def action(self,state):
		Q = [np.dot(self.w, self.X(state,a)) for a in range(self.nA)]
		# print("THIS STATE " , state, Q, np.argmax(Q))
		return np.argmax(Q)

	def restore_fromfile(self, filename):
		with open(filename, 'rb') as f:
			self.w = np.load(f)