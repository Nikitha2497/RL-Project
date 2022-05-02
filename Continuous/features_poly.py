#This implements the polynomial feature extraction

from features import StateActionFeatureVector

from features import StateFeatureVector

import numpy as np


#State is of the form(s1, s2) and the features for each action are [1, s1, s2, s1s2]
#We have used the similar interface from the assignment.

class StateActionFeatureVectorWithPoly(StateActionFeatureVector):
    def __init__(self,
                 num_actions:int):
        """
        num_actions: the number of possible actions
        """
        
        self.num_actions = num_actions
        self.num_dimesions_per_action = 4
        self.dimension = self.num_dimesions_per_action*self.num_actions

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_dimensions_per_action
        """
        return self.dimension

   
    def __call__(self, s, done, a) -> np.array:
    
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        ret_array = np.zeros(self.dimension)
        
        if done:
            return ret_array

        
        ret_array[a] = 1
        ret_array[a+1] = s[0]
        ret_array[a+2] = s[1]
        ret_array[a+3] = s[0]*s[1]

        return ret_array


#State is of the form(s1, s2) and the features for each state are [1, s1, s2, s1s2]
#We have used the similar interface from the assignment.


class StateFeatureVectorWithPoly(StateFeatureVector):
    def __init__(self):

        self.dimension = 4

    def feature_vector_len(self) -> int:
        return self.dimension

   
    def __call__(self, s, done) -> np.array:
    
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        ret_array = np.zeros(self.dimension)
        
        if done:
            return ret_array

        
        ret_array[0] = 1
        ret_array[1] = s[0]
        ret_array[2] = s[1]
        ret_array[3] = s[0]*s[1]

        return ret_array

