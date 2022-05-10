import numpy as np


#Base class for state action feature vector
#We have used a similar interface from the assignment.
class StateActionFeatureVector():
	def __init__(self,
				 num_actions:int):
		#TODO - Implement if anything is common here.
		raise NotImplementedError()

	def feature_vector_len(self) -> int:
		raise NotImplementedError()
	def __call__(self, s, a) -> np.array:
		raise NotImplementedError()

#Base class for state feature vector
class StateFeatureVector():
	def __init__(self):
		#TODO - Implement if anything is common here.
		raise NotImplementedError()
	def feature_vector_len(self) -> int:
		raise NotImplementedError()   
	def __call__(self, s) -> np.array:
		raise NotImplementedError()

