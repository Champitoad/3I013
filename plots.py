import pandas as pd
import matplotlib.pyplot as plt
from consts import move_dists

def load_movedist_data(move_dists):
    dfs = {}
    for md in move_dists:
        dfs[md] = pd.read_csv('results/move_distance_{}.csv'.format(md), index_col=0)
    return dfs

def stats(dfs, stat, attrs):
    stats = pd.DataFrame(columns=attrs)
    for i, df in dfs.items():
        stats.loc[i] = df[attrs].describe().loc[stat]
    return stats

if __name__ == '__main__':
    dfs = load_movedist_data(move_dists)

    acc_stats = pd.DataFrame(index=move_dists, columns=['min', 'mean', 'max'])
    acc_stats['min'] = stats(dfs, 'min', ['accuracy'])
    acc_stats['mean'] = stats(dfs, 'mean', ['accuracy'])
    acc_stats['max'] = stats(dfs, 'max', ['accuracy'])

    fig, ax = plt.subplots(figsize=(10,7))
    linewidths = [1.5, 2.5, 1.5]
    styles = ['--', 'o-', '--']
    markersizes = [4, 6, 4]
    labels = ['Minimum', 'Moyenne', 'Maximum']
    for col, lw, style, ms, label in zip(acc_stats.columns, linewidths, styles, markersizes, labels):
        acc_stats[col].plot.line(ax=ax, lw=lw, style=style, ms=ms, label=label)
    plt.xticks(move_dists)
    plt.xlabel("Distance de déplacement")
    plt.ylabel("Accuracy")
    plt.ylim((0,100))
    plt.legend()
    plt.savefig('rapport/final/img/movedist_acc_mean.png')

    acc_std = stats(dfs, 'std', ['accuracy'])
    acc_std.columns = ['Variance']

    fig, ax = plt.subplots(figsize=(8,6))
    acc_std.plot.line(ax=ax, lw=2.5, style='o-', ms=6)
    plt.xticks(move_dists)
    plt.xlabel("Distance de déplacement")
    plt.ylabel("Accuracy")
    plt.ylim((0,15))
    plt.savefig('rapport/final/img/movedist_acc_std.png')
