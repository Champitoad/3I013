import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing.pool import ThreadPool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter
from consts import *

grid_size = (1,1000)
num_episodes = 500

def train(digit):
    global mean_accs
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={digit}, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid, move_distance)
    pred = Predicter(cursor_size)
    mean_acc = pred.learn(numgrid, num_episodes, directions)
    path = pred.save_model("models/predicter{}.ckpt".format(digit))
    print("Predicter {} trained with mean accuracy of {}%".format(digit, mean_acc*100))
    mean_accs[digit] = mean_acc

if __name__ == '__main__':
    mean_accs = np.zeros(10)
    with ThreadPool(10) as p:
        p.map(train, range(10))
    mean_accs.tofile('train_mean_accs', sep=', ')
