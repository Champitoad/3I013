import numpy as np

red = '\033[91m'
green = '\033[32m'
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

    def get_trajectory(self, render=False):
        """
        Run agent-environment loop for one whole episode (trajectory).
        Returns the dictionary of results.
        """
        observation = self.env.reset()
        observations = []
        actions = []
        rewards = []
        for i in range(self.config['episode_max_length']):
            observations.append(observation)
            action = self.act(observation)
            actions.append(action)
            observation, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            color = green if reward >= 0 else red
            print(color + 'action:', action, endc)
            if done:
                break
            if render:
                self.env.render()
        return {'reward': np.array(rewards),
                'observation': np.array(observations),
                'action': np.array(actions),
                'done': done,
                'steps': i + 1
                }

    def learn(self):
        raise NotImplementedError
