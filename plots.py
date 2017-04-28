import pandas as pd
import matplotlib.pyplot as plt
from consts import move_dists

def load_movedist_data(move_dists):
    dfs = {}
    for md in move_dists:
        dfs[md] = pd.read_csv('results/move_distance_{}.csv'.format(md), index_col=0)
    return dfs

def plot(dfs, stat, attrs):
    stats = pd.DataFrame(columns=attrs)
    for i, df in dfs.items():
        stats.loc[i] = df[attrs].describe().loc[stat]
    return stats.plot

if __name__ == '__main__':
    dfs = load_movedist_data(move_dists)
    plot(dfs, 'mean', ['score']).bar(color='#f57c00', legend=False)
    plt.axhline(0, color='0.4', linewidth=0.75)
    plt.xlabel("Distance de déplacement")
    plt.ylabel("Score")
    plt.show()
