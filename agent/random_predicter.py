import sys
sys.path.append("..")

import random
import numpy as np
import tensorflow as tf
from multiprocessing.pool import ThreadPool

from agent import Agent
from autoencoder.predicter import Predicter
from consts import directions

class RandomPredicterAgent(Agent):
    """
    Agent for NumGrid moving in random directions, and labelling
    using 10 predicters (one for each digit).
    """
    def __init__(self, numgrid):
        super().__init__(numgrid, episode_max_length=numgrid.num_steps)

        self.preds = []
        for digit in range(10):
            with tf.Graph().as_default():
                pred = Predicter(numgrid.cursor_size)
                pred.load_model("models/predicter{}.ckpt".format(digit))
            self.preds.append(pred)

        self.image = None

    def act(self, observation):
        direction = random.choice(tuple(directions))
        digit = 10

        next_image = Predicter.normalize(observation)
        if self.image is None:
            self.image = next_image
            return (digit, direction)

        accuracy = lambda pred: pred.accuracy(self.image, direction, next_image)
        with ThreadPool(10) as p:
            accs = p.map(accuracy, self.preds)
        # print(accs)
        accs = np.array(accs)
        accs = accs[accs > 0.80]
        if len(accs) > 0:
            digit = np.argmax(accs)

        return (digit, direction)
