import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

grid_size = (5,5)
num_episodes = 10
num_exp = 1
acc_thr = [98.5,88,85,87,88,99.3,95,99.4,96,87.6]
score_thr = 5

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    agent = RandomPredicterAgent(numgrid, acc_thr, score_thr)
    score = 0
    num_preds = 0
    num_ok = 0
    for episode in range(num_episodes):
        agent.score = np.zeros(10)
        traj = agent.get_trajectory(print_actions=False, render=False, move_distance=move_distance)
        for i in range(num_steps):
            rew = traj['reward'][i]
            if rew != 0:
                score += rew
                num_preds += 1
                if rew > 0:
                    num_ok += 1
    accuracy = num_ok / num_preds * 100
    print("\n********* EXPERIENCE", num, "**********")
    print("num de predictions : ", num_preds)
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
print(score_thr)
