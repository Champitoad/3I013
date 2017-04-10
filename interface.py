import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
import time
from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter

numgrid = NumGrid(size=(5,5), cursor_size=(12,12), digits={0}, num_steps=100)
env = DiscreteDirectionWrapper(numgrid)

pred = Predicter(numgrid.cursor_size)
pred.load_model("models/predicter0.ckpt")

mean_acc = 0
total_mean_acc = 0
for i_episode in range(100):
    observation = env.reset()
    reward = None
    done = False
    info = {}
    image = observation.reshape(1,-1).astype(np.float32) / 255
    while not done:
        action = (10, env.direction_space.sample())
        observation, reward, done, info = env.step(action)
        next_image = observation.reshape(1,-1).astype(np.float32) / 255
        mean_acc += pred.accuracy(image, action[1], next_image)
        image = next_image
    mean_acc /= numgrid.num_steps
    total_mean_acc += mean_acc
total_mean_acc /= 100
print("Total mean accuracy: {}".format(total_mean_acc))
