import sys
sys.path.append("..")

import random
import numpy as np
import tensorflow as tf
from multiprocessing.pool import ThreadPool

from agent import Agent
from autoencoder.predicter import Predicter
from consts import *

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
        for digit in range(1):
            with tf.Graph().as_default():
                print("on charge un nouveaux model")
                pred = Predicter(numgrid.cursor_size)
                pred.load_model("models/predicter2.0{}.ckpt".format(digit))
                print("charge")
            self.preds.append(pred)

        self.image = None
        self.acc_thr = acc_thr
        self.score_thr = score_thr
        self.score = np.zeros(10) # Cumulated score for each predicter

    def circuit(self):
        self.cirQ=[]
        d=0
        for i in range(num_steps):
            for j in (range(d+1)):
                if i%4==0:
                    self.cirQ.append(3)
                elif i%4==1:
                    self.cirQ.append(0)
                elif i%4==2:
                    self.cirQ.append(2)
                else:
                    self.cirQ.append(1)
            d+=1


    def act(self, observation):
        digit = 10
        next_image = Predicter.normalize(observation)
        self.direction =random.choice(tuple(directions))
        if self.image is None:
            self.image = next_image 
            return ((digit, self.direction), 0)

        accuracy = lambda pred: pred.accuracy(self.image, self.direction, next_image)

        with ThreadPool(1) as p:
            accs = p.map(accuracy, self.preds)
        
        #self.cirQ.pop(0)

        #self.score += np.array(accs) - np.array(self.acc_thr) #********************

        #prediction = self.score[self.score >= self.score_thr]
        #print(np.round(self.score, 3))
        # if np.max(self.score)>=self.score_thr:
        #     digit = np.argmax(self.score)
        #     print("on predit ", digit, "avec un score de ", np.max(self.score), "\n")
        #     self.score = np.zeros(10)
        #     print(self.score)
        #     self.env.reset()

        
        # if len(prediction) > 0:
        #     digit = np.argmax(prediction)
        #     print("on predit ", digit, "avec un score de ", np.max(prediction), "\n")
        #     self.score = np.zeros(10)
            

        self.image = next_image
        
        return ((digit, self.direction), accs[0])

    def close(self):
        self.preds[0].close()
