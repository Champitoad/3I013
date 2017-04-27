import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

grid_size = (1,15)
num_episodes = 100
num_exp = 10
acc_thr = [0.8599,0.9500,0.8637,0.8600,0.9073,0.8598,0.8790,0.8990,0.8780,0.8979]#[0.97,0.98,0.95,0.96,0.97,0.97,0.97,0.98,0.96,0.97]
score_thr = 1

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps, mnist_images_path='t10k-images-idx3-ubyte.gz', mnist_labels_path='t10k-labels-idx1-ubyte.gz')
    numgrid = DiscreteDirectionWrapper(numgrid, move_distance)
    agent = RandomPredicterAgent(numgrid, acc_thr, score_thr)
    score = 0
    num_preds = 0
    num_ok = 0
    for episode in range(num_episodes):
        agent.score = np.zeros(10)
        print(agent.score)
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
