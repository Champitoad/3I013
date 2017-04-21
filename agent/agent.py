import numpy as np
from consts import *

red = '\033[91m'
green = '\033[32m'
yellow = '\033[93m'
endc = '\033[0m'

class Agent:
    """
    Abstract class defining an agent properties and interface.
    """
    def __init__(self, env, **usercfg):
        self.env = env
        self.config = {'episode_max_length': 10**6}
        self.config.update(usercfg)

    def act(self, observation):
        raise NotImplementedError

    def get_trajectory(self, print_actions=False, render=False, move_distance=move_distance):
        """
        Run agent-environment loop for one whole episode (trajectory).
        Returns the dictionary of results.
        """
        self.circuit()
        observation = self.env.reset()
        observations = []
        actions = []
        rewards = []
        for i in range(self.config['episode_max_length']):
            observations.append(observation)
            if i%move_distance==0:
                pred=True
            else:
                pred=False
            action = self.act(observation, pred)
            actions.append(action)
            observation, reward, done, info = self.env.step(action)
            if pred:
                print("on est sur un ", info["digit"])
            rewards.append(reward)
            if print_actions:
                if reward == 0:
                    color = yellow
                else:
                    color = green if reward > 0 else red
                print(color + 'action:', action, endc)
            if render:
                self.env.render()
            if done:
                break
        return {'reward': np.array(rewards),
                'observation': np.array(observations),
                'action': np.array(actions),
                'done': done,
                'steps': i + 1
                }
                

    def learn(self):
        raise NotImplementedError
