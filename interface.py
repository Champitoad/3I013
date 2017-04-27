import sys
sys.path.append("gym-numgrid")
sys.path.append("agent")

import numpy as np
import pandas as pd
from multiprocessing import Pool

from gym_numgrid.envs import NumGrid
from gym_numgrid.wrappers import *
from random_predicter import RandomPredicterAgent
from consts import *

mnist_images_path = 'mnist/t10k-images-idx3-ubyte.gz'
mnist_labels_path = 'mnist/t10k-labels-idx1-ubyte.gz'

grid_size = (1,50)
num_episodes = 50
num_exp = 100
score_thr = 5

def experience(num):
    numgrid = NumGrid(size=grid_size, cursor_size=cursor_size, num_steps=num_steps,\
                      mnist_images_path=mnist_images_path, mnist_labels_path=mnist_labels_path)
    numgrid = DiscreteDirectionWrapper(numgrid, move_distance)
    agent = RandomPredicterAgent(numgrid, score_thr)
    score = 0
    num_preds = 0
    num_ok = 0
    for episode in range(1, num_episodes+1):
        traj = agent.get_trajectory(print_actions=False, render=False)
        for i in range(num_steps):
            rew = traj['reward'][i]
            if rew != 0:
                score += rew
                num_preds += 1
                if rew > 0:
                    num_ok += 1
        print("Episodes done: {}/{}".format(episode, num_episodes))
    accuracy = num_ok / num_preds * 100
    pred_rate = num_preds / (num_steps * num_episodes) * 100
    print("\n********* EXPERIENCE", num, "**********")
    print("Score: {}".format(score))
    print("Accuracy: {:.4g}%".format(accuracy))
    print("Prediction rate: {:.4g}%".format(pred_rate))
    return (score, accuracy, pred_rate)

def experiment(move_distance):
    print('\n************************ MOVE_DISTANCE = {} ********************'.format(move_distance))

    with Pool(num_exp) as p:
        results = p.map(experience, range(1, num_exp+1))
    scores, accs, pred_rates = tuple(zip(*results))

    data = pd.DataFrame(columns=['score', 'accuracy', 'pred_rate'])
    data['score'] = scores
    data['accuracy'] = accs
    data['pred_rate'] = pred_rates

    stats = data.describe().T
    mean_score = stats['mean']['score']
    mean_acc = stats['mean']['accuracy']
    mean_pred_rate = stats['mean']['pred_rate']

    max_score = 3 * num_steps * num_episodes
    print("\n********* FINAL RESULTS **********")
    print("Score range: [{};{}]".format(-max_score, max_score))
    print("Mean score: {:.4g}".format(mean_score))
    print("Mean accuracy: {:.4g}%".format(mean_acc))
    print("Mean prediction rate: {:.4g}%".format(mean_pred_rate))

    return data

if __name__ == '__main__':
    for move_distance in (1,4,8,12):
        results = experiment(move_distance)
        results.to_csv('results/move_distance_{}.csv'.format(move_distance))
