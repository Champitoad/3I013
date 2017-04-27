import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter
from consts import *



def train(digit):
    grid_size = (1,500)
    num_episodes = 1000
    move_distance = 1

    print("table: ", grid_size)
    print("num iterations: ", num_episodes*num_steps)
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={digit}, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    pred = Predicter(cursor_size)
    m=pred.learn(numgrid, num_episodes, directions, move_distance)
    path = pred.save_model("models/predicter{}.ckpt".format(digit))
    print("Predicter {} saved in file: {}".format(digit, path))
    print("moyenne du digit {}", format(digit), "est de ", m)

with Pool(10) as p:
	print("go")
	p.map(train, range(10))
