# Process data and statistics
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

setting = 'lr0.01_wd0.001_dropout0.0_step1_batch64'

FILE = ['out_femnist_' + x for x in ['0.2']]

files = []

metrics = {
    x: []
    for x in [
        'train_avg_loss', 'val_avg_loss', 'test_avg_loss', 'train_acc',
        'val_acc', 'test_acc', 'train_f1', 'val_f1', 'test_f1'
    ]
}

for p in FILE:
    os.chdir(os.path.join('FederatedScope', p))
    trail_names = [x for x in os.listdir() if setting in x]
    print(trail_names)
    for seed, path in enumerate(trail_names):
        with open(os.path.join(path, 'eval_results.log')) as f:
            F = f.readlines()

            for idx, line in enumerate(F):
                results = eval(line)
                for key in metrics:
                    if idx == 0:
                        metrics[key].append([])
                    if 'Results_weighted_avg' not in results:
                        continue
                    metrics[key][seed].append(
                        results['Results_weighted_avg'][key])

    os.chdir('../..')
    
# Draw

FONTSIZE = 60
MARKSIZE = 15
ROUND = 500
GAP = 2
legend = []

for e, metric in enumerate(metrics):
    legend.append(metric)
    if e % len(['train', 'val', 'test']) == 0:
        plt.figure(figsize=(40, 15))
        # plt.title(f'{}', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)

        plt.xlabel('Round', size=FONTSIZE)
        plt.ylabel(f'{metric[6:]}', size=FONTSIZE)

    data = metrics[metric]

    # Check the length
    for i in data:
        i += ([i[-1]] * ((ROUND - GAP * len(i)) // GAP))

    plt.errorbar(
        np.arange(0, ROUND, GAP),
        np.mean(data, axis=0),
        yerr=np.std(data, axis=0),
        #fmt='',
        marker='*',
        markersize=MARKSIZE,
        elinewidth=2,
        capsize=5,
        capthick=5,
        #color='red',
        alpha=0.5)

    if e % len(['train', 'val', 'test']) == 2:
        plt.legend(legend, fontsize=FONTSIZE)
        legend = []
        plt.show()
        plt.close()