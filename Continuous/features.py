
#Base class for state action feature vector
class StateActionFeatureVector():
    def __init__(self,
                 num_actions:int):
    	#TODO - Implement if anything is common here.


    def feature_vector_len(self) -> int:
        raise NotImplementedError()

   
    def __call__(self, s, done, a) -> np.array:
       	raise NotImplementedError()
