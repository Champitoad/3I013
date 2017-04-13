import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter
from consts import *

grid_size = (50,50)
num_steps = 200
num_episodes = 20
move_distance = 10

def train(digit):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={digit}, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    pred = Predicter(cursor_size)
    pred.learn(numgrid, num_episodes, directions, move_distance)
    path = pred.save_model("models/predicter{}.ckpt".format(digit))
    print("Predicter {} saved in file: {}".format(digit, path))

with Pool(10) as p:
    p.map(train, list(range(10)))
