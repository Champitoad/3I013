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
    
    The agent cumulates evidence from the predicters before trying a guess of the digit.
    """
    def __init__(self, numgrid, acc_thr, score_thr):
        """
        numgrid -- NumGrid environment to evaluate
        acc_thr -- decision threshold for the predicters score at each step
        score_thr -- decision threshold to choose which predicter is right
        """
        super().__init__(numgrid, episode_max_length=numgrid.num_steps)

        # Load predicter models from disk
        self.preds = []
        for digit in range(10):
            with tf.Graph().as_default():
                pred = Predicter(numgrid.cursor_size)
                pred.load_model("models/predicter{}.ckpt".format(digit))
            self.preds.append(pred)

        self.image = None
        self.acc_thr = acc_thr
        self.score_thr = score_thr
        self.score = np.zeros(10) # Cumulated score for each predicter

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
        self.score += np.array(accs) - self.acc_thr
        prediction = self.score[self.score >= self.score_thr]
        print(np.round(self.score, 3))
        if len(prediction) > 0:
            digit = np.argmax(prediction)
            self.score = np.zeros(10)
            print()

        self.image = next_image
        return (digit, direction)
