# RL-Project
RL Course Project - Chance Constraint Motion Planning
Authors - Apurva Patil, Nikitha Gollamudi

Required python modules - numpy, seaborn, typing, matlabplot, pandas

The folder structure of the code base is as follows - 
RL-Project
---> Discrete
---> Continuous

Common interfaces like env.py, policy.py are present in the RL-Project folder.
Common util classes like plot.py is present in the RL-Project folder.


We have three run files each for discrete and continuous state space that generates results. These are described below - 


DISCRETE - 
Run 'python3 test_discrete.py'  - This runs the modified Q-learning algorithm for 100,000 episodes for 10 runs. The discrete env, parameters for the tuning can be set in this file. This stores the pi_star in "./data/" folder and the q_star for the start state plots are stored in "./results/" folder. Currently, this generates results for two etas - 15 and 50.

Run 'python3 test_pfail_prediction.py' - This runs the modified TD(0) algorithm for 100,000 episodes for 10 runs. This restores the pi_star from './data/' folder and plots the p_Fail graphs. Currently, this generates results for two etas - 15 and 50. The results are stored in './results/' folder

Run 'python3 plot_discrete.py' - his restores the pi_star from './data/' folder and plots the grid-world, policy, no-noise trajectory graphs. Currently, this generates results for two etas - 15 and 50. The results are stored in './results/' folder


CONTINUOUS - 
Run 'python3 test_continous.py'  - This runs the modified Semi gradient Sarsa algorithm for 100,000 episodes for 10 runs. The continuous env, parameters for the tuning can be set in this file. This stores the pi_star in "./data/" folder and the q_star for the start state plots are stored in "./results/" folder. Currently, this generates results for two etas - 15 and 70.

Run 'python3 test_pfail_prediction_continuous.py' - This runs the modified semi gradient TD(0) algorithm for 100,000 episodes for 10 runs. This restores the pi_star from './data/' folder and plots the p_Fail graphs. Currently, this generates results for two etas - 15 and 70. The results are stored in './results/' folder

Run 'python3 plot_continuous.py' - his restores the pi_star from './data/' folder and plots the domain, no-noise trajectory and sample trajectories graphs. Currently, this generates results for two etas - 15 and 70. The results are stored in './results/' folder



