import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool
from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter

grid_size = (10,10)
cursor_size = (12,12)
num_steps = 100
num_episodes = 100

def train(digit):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={digit}, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    pred = Predicter(cursor_size)
    pred.learn(numgrid, num_episodes)
    pred.save_model("models/predicter{}.ckpt".format(digit))

with Pool(10) as p:
    p.map(train, list(range(10)))
