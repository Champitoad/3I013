import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
import time
from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from autoencoder.predicter import Predicter

yellow = '\033[93m'
endc = '\033[0m'

numgrid = NumGrid(size=(1,20), cursor_size=(10,10), digits={3})
env = DirectionWrapper(numgrid)
env.configure(num_steps=500, render_scale=1.5)

agent = Predicter(learning_rate=0.001, nbp_input=np.prod(numgrid.cursor_size), time_training=1000)

for i_episode in range(5):
    print("\n********* EPISODE", i_episode, "**********\n")
    env.env.cursor_pos = np.array((10,0))
    env.env.steps = 0
    observation = None
    reward = None
    done = False
    info = {"cursor": env.env.cursor}
    action = agent.act(observation, reward, done, info)
    while not done:
        env.render()
        observation, reward, done, info = env.step(action)
        if info["out_of_bounds"]:
            print(yellow + "Can't get out of the world!" + endc)
        action = agent.act(observation, reward, done, info)
        # time.sleep(0.01)
