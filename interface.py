import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

grid_size = (1,10)
num_steps = 40
num_episodes = 10

numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps)
numgrid = DiscreteDirectionWrapper(numgrid)

agent = RandomPredicterAgent(numgrid)

score = 0
for episode in range(num_episodes):
    score += agent.get_trajectory(render=True)['reward'].sum()
max_score = 3 * num_steps * num_episodes
accuracy = (score + max_score) / (2 * max_score)
print("Score: {}".format(score))
print("Accuracy: {}%".format(accuracy*100))
