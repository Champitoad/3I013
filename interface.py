from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *

from autoencoder.predicter import Predicter

env = NumGrid(size=(1,1000), digits={1})
env.configure(num_steps=50)

agent = Predicter(learning_rate=0.1, nbp_input=784, time_training=1000)

for i_episode in range(10):
    print("\n********* EPISODE", i_episode, "**********\n")
    observation = env.reset()
    reward = 0
    done = False
    info = {'cursor': env.cursor}
    while not done:
        action = agent.act(observation, reward, done, info)
        observation, reward, done, info = env.step(action)
        if info['out_of_bounds']:
            print(yellow + "Can't get out of the world!" + endc)
