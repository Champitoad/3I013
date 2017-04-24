import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter
from consts import *

#grid_size = (1,1000)
#num_episodes = 100
move_distance = 1


def train(digit):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, digits={digit}, num_steps=num_steps)
    numgrid = DiscreteDirectionWrapper(numgrid)
    pred = Predicter(cursor_size)
    m=pred.learn(numgrid, num_episodes, directions, move_distance)
    path = pred.save_model("models/predicter2.0{}.ckpt".format(digit))
    print("Predicter {} saved in file: {}".format(digit, path))
    print("accuracy moyenne de {}", format(digit), "est de ", m)

def act(num):
    num_episodes=100
    num_steps=100
    grid_size = (50,50)
    print("nb neurone", nb_neurone)
    print("num iterations: ", num_episodes*num_steps)
    train(0)
