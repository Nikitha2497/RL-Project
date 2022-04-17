import numpy as np

from env_discrete import DiscreteEnv
from Q_learning_discrete import QLearning

env = DiscreteEnv()


initQ = np.zeros((env.nS, env.nA))

gamma = 1
alpha = 0.1
num_episode = 1000
eta = 1
epsilon = .0

(Q_star, pi_star) = QLearning(env, 
    gamma,
    alpha,
    num_episode,
    eta,
    initQ,
    epsilon)


print(Q_star)
pi_star.print_all()



