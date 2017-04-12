import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

grid_size = (1,10)
num_steps = 20
num_episodes = 10

numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps)
numgrid = DiscreteDirectionWrapper(numgrid)

agent = RandomPredicterAgent(numgrid, score_thr=5)

score = 0
num_preds = 0
num_ok = 0
for episode in range(num_episodes):
    traj = agent.get_trajectory(render=True)
    for i in range(num_steps):
        rew = traj['reward'][i]
        if rew != 0:
            score += rew
            num_preds += 1
            if rew > 0:
                num_ok += 1
max_score = 3 * num_steps * num_episodes
accuracy = num_ok / num_preds * 100
print("Score range: [{};{}]".format(-max_score, max_score))
print("Final score: {}".format(score))
print("Accuracy: {:.4g}%".format(accuracy))
