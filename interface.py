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
num_exp = 1
acc_thr = [98.5,88,85,87,88,99.3,95,99.4,96,87.6]
score_thr = 5
x=[10,20,30,40,50,60,70,80,90,100]
y=[0.902,0.870,0.907, 0.885,0.916,0.853,0.855,0.873,0.804,0.863]
y2=[0.905,0.890,0.865,0.863,0.880,0.926,0.892,0.844,0.888,0.879]
y3=[0.884,0.846,0.846,0.870,0.877,0.877,0.912,0.816,0.884,0.927]
y4=[0.889,0.862,0.830,0.870,0.862,0.910,0.826,0.885,0.880,0.842]
n=1

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={0}, num_steps=num_steps)
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
    #accuracy = num_ok / num_preds * 100
    print("\n********* EXPERIENCE", num, "**********")
    #print("num de predictions : ", num_preds)
    print("Score: {}".format(score))
    #print("Accuracy: {:.4g}%".format(accuracy))
    accuracy=accuracy/num_episodes
    print(accuracy)
    return (accuracy)


# results=[]
# for n in range(1,10):
#     grid_size=(50, 50)
#     ts.act(n*0.1)
#     num_episodes=10
#     num_steps=10
#     results.append(experience(1))
#     print(results)
ts.act(n)
experience(1)
# plt.scatter(x, y)
# plt.show()
# plt.savefig('nb_neurone.png')

#scores, accs = tuple(zip(*results))

max_score = 3 * num_steps * num_episodes
# print("\n********* FINAL RESULTS **********")
# print("Score range: [{};{}]".format(-max_score, max_score))
# print("Mean score: {:.4g}".format(np.mean(scores)))
# print("Mean accuracy: {:.4g}%".format(np.mean(accs)))
# print(score_thr)
