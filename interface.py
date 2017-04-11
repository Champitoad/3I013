import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent

grid_size = (10,10)
cursor_size = (12,12)
num_steps = 100
num_episodes = 10

numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps)
numgrid = DiscreteDirectionWrapper(numgrid)

agent = RandomPredicterAgent(numgrid)

score = 0
for i in range(num_episodes):
    score += agent.get_trajectory(render=False)['reward'].sum()
print("Final score: {}".format(score))
