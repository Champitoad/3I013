import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter

numgrid = NumGrid(size=(5,5), cursor_size=(8,8), digits={0}, num_steps=100)
env = DiscreteDirectionWrapper(numgrid)

agent = Predicter(learning_rate=0.001, nbp_input=np.prod(numgrid.cursor_size), move_distance=10)

for i_episode in range(100):
    observation = env.reset()
    reward = None
    done = False
    info = {}
    action = agent.act(observation, reward, done, info)
    while not done:
        observation, reward, done, info = env.step(action)
        action = agent.act(observation, reward, done, info)

agent.save_model("models/predicter0.ckpt")
