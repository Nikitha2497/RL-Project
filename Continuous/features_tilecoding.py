#This implements the tilecoding feature coding

from features import StateActionFeatureVector
from features import StateFeatureVector
import numpy as np
import math


#We have used the same implementation we used in the assignment 4
class StateActionFeatureVectorWithTile(StateActionFeatureVector):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """

        self.num_tilings = num_tilings
        self.num_tiles = 1
        self.num_actions = num_actions
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width
        self.offset = {}
        self.num_tiles_dimension = []
        for dimension in range(0, len(state_low)):
            self.num_tiles *=  math.ceil((state_high[dimension] - state_low[dimension])/float(tile_width[dimension])) + 1
            self.num_tiles_dimension.append( math.ceil((state_high[dimension] - state_low[dimension])/float(tile_width[dimension])) + 1)

        product = 1

        for i in range(0, len(self.num_tiles_dimension) - 1):
            product *= self.num_tiles_dimension[i]

        for i in range(0, self.num_tilings):
            offset_dimension  = []
            for dimension in range(0, len(state_low)):
                offset_dimension.append((i*tile_width[dimension])/float(num_tilings))

            self.offset[i] = offset_dimension

        self.dimension = self.num_actions * self.num_tilings * self.num_tiles
        print("num_tilings", self.num_tilings)


    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        
        print("dimension" , self.dimension)
        return self.dimension

   
    def __call__(self, s, a) -> np.array:
    
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        ret_array = np.zeros(self.dimension)

        shape = []
        shape.append(self.num_actions)
        shape.append(self.num_tilings)

        for i in range(0, len(self.num_tiles_dimension)):
            shape.append(self.num_tiles_dimension[i])

        ret_nd_array = np.zeros(shape)

        
        for tiling in range(0, self.num_tilings):
            state_rep = []
            state_rep.append(a)        
            state_rep.append(tiling)
            offset_tiling = self.offset[tiling]
        
            for dimension in range(0,len(s)):
                val = s[dimension] + offset_tiling[dimension] - self.state_low[dimension]
                
                i = math.floor(val/float(self.tile_width[dimension]))
                state_rep.append(i)
            ret_nd_array[tuple(state_rep)] = 1

        ret_array = ret_nd_array.flatten()
        # print(len(ret_array))
        return ret_array


class StateFeatureVectorWithTile(StateFeatureVector):

    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """

        self.dimensions = len(state_low)
        self.num_tilings  = num_tilings
        self.offset = {}
        self.tile_width = tile_width
        self.state_low = state_low
        self.state_high = state_high

        self.num_tiles = 1
        for dimension in range(0, self.dimensions):
        	self.num_tiles *=  math.ceil((state_high[dimension] - state_low[dimension])/float(tile_width[dimension])) + 1

        for i in range(0, self.num_tilings):
            self.num_tiles_dimension = []
            offset_dimension  = []
            for dimension in range(0, self.dimensions):
                self.num_tiles_dimension.append( math.ceil((state_high[dimension] - state_low[dimension])/float(tile_width[dimension])) + 1)
                offset_dimension.append((i*tile_width[dimension])/float(num_tilings))

            self.offset[i] = offset_dimension

        self.dimension = self.num_tilings * self.num_tiles

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_tilings * num_tiles
        """
        return self.dimension


    def __call__(self,s):

        ret_array = np.zeros(self.dimension)

        shape = []
        shape.append(self.num_tilings)

        for i in range(0, len(self.num_tiles_dimension)):
            shape.append(self.num_tiles_dimension[i])

        ret_nd_array = np.zeros(shape)

        
        for tiling in range(0, self.num_tilings):
            state_rep = []        
            state_rep.append(tiling)
            offset_tiling = self.offset[tiling]
        
            for dimension in range(0,len(s)):
                val = s[dimension] + offset_tiling[dimension] - self.state_low[dimension]
                
                i = math.floor(val/float(self.tile_width[dimension]))
                state_rep.append(i)
            ret_nd_array[tuple(state_rep)] = 1

        ret_array = ret_nd_array.flatten()
        return ret_array

