import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *
import train_subnet as ts
from matplotlib import pyplot as plt

grid_size = (5,5)
num_episodes = 10
num_exp = 50
acc_thr = [98.5,88,85,87,88,99.3,95,99.4,96,87.6]
score_thr = 5
x=[10,20,30,40,50,60,70,80,90,100]
y=[0.902,0.870,0.907, 0.885,0.916,0.853,0.855,0.873,0.804,0.863]
y2=[0.905,0.890,0.865,0.863,0.880,0.926,0.892,0.844,0.888,0.879]
y3=[0.884,0.846,0.846,0.870,0.877,0.877,0.912,0.816,0.884,0.927]
y4=[0.889,0.862,0.830,0.870,0.862,0.910,0.826,0.885,0.880,0.842]
n=0.7

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={0}, num_steps=num_steps, mnist_images_path='t10k-images-idx3-ubyte.gz', mnist_labels_path='t10k-labels-idx1-ubyte.gz')
    numgrid = DiscreteDirectionWrapper(numgrid)
    agent = RandomPredicterAgent(numgrid, acc_thr, score_thr)
    score = 0
    num_preds = 0
    num_ok = 0
    accuracy=0
    for episode in range(num_episodes):
        agent.score = np.zeros(10)
        traj = agent.get_trajectory(print_actions=False, render=False, move_distance=move_distance)
        accuracy+=traj['acuracy']
        for i in range(num_steps):
            rew = traj['reward'][i]
            if rew != 0:
                score += rew
                num_preds += 1
                if rew > 0:
                    num_ok += 1
    agent.close()
    print("\n********* EXPERIENCE", num, "**********\n")
    accuracy=accuracy/num_episodes
    print(accuracy, "\n")
    return accuracy

# acc=0
# for i in range(3):
#     print(i)
#     ts.act(n)
#     acc+=experience(1)
# acc=acc/50
# print(acc)

with Pool(num_exp) as p:
    entrainement=p.map(ts.act, range(1, num_exp+1))
    results = p.map(experience, range(1, num_exp+1))
    print("Mean accuracy: {:.4g}%".format(np.mean(results)))
    
#accs = tuple(zip(*results))

# max_score = 3 * num_steps * num_episodes
print("\n********* FINAL RESULTS **********")
# print("Score range: [{};{}]".format(-max_score, max_score))
# print("Mean score: {:.4g}".format(np.mean(scores)))
#print("Mean accuracy: {:.4g}%".format(np.mean(results)))
# print(score_thr)


