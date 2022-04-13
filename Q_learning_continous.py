import numpy as np
import math

from feature import StateActionFeatureVector

#We are using linear function approximation. 
def QLearning(
    env, # openai gym environment (Assuming we can create custom environment in open ai otherwise will have to replace this with custom class)
    gamma:float, # discount factor
    alpha:float, # step size
    X:StateActionFeatureVector,
    num_episode:int,
    eta: float,
    epsilon=.0
) -> np.array:

    w = np.zeros((X.feature_vector_len()))


    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        #print(len(w))
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)


    for i in range(0, num_episode):
        #print(i)
        state = env.reset()
        done = False

        
        while True:
            action = epsilon_greedy_policy(new_state, done, w, epsilon)
            new_state, reward, done, goal, info = env.step(action)

            x = X(state, done, action) #features

            Q = np.dot(w, x)

            #How do we get if the state is the goal state or other terminal state?

            #How do we min in case of function approximation?
            
            # delta = reward + gamma*

            state = new_state

            if done:
                break

    return w
