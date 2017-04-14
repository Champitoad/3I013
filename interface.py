import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

grid_size = (1,1000)
num_episodes = 10
num_exp = 5
score_thr = 5

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    agent = RandomPredicterAgent(numgrid, score_thr)
    score = 0
    num_preds = 0
    num_ok = 0
    for episode in range(num_episodes):
        traj = agent.get_trajectory(print_actions=False, render=False)
        for i in range(num_steps):
            rew = traj['reward'][i]
            if rew != 0:
                score += rew
                num_preds += 1
                if rew > 0:
                    num_ok += 1
    accuracy = num_ok / num_preds * 100
    print("\n********* EXPERIENCE", num, "**********")
    print("Score: {}".format(score))
    print("Accuracy: {:.4g}%".format(accuracy))
    return (score, accuracy)

with Pool(num_exp) as p:
    results = p.map(experience, range(1, num_exp+1))
scores, accs = tuple(zip(*results))

max_score = 3 * num_steps * num_episodes
print("\n********* FINAL RESULTS **********")
print("Score range: [{};{}]".format(-max_score, max_score))
print("Mean score: {:.4g}".format(np.mean(scores)))
print("Mean accuracy: {:.4g}%".format(np.mean(accs)))
