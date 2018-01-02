from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa/deep_sa_batch_interval_study.csv')

df = df.sort_values(['batch_interval', 'train_batch_size'])


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):

    ax = ax if ax is not None else plt.gca()

    if np.isscalar(yerr) or len(yerr) == len(y):

        ymin = [y_i - yerr_i for (y_i, yerr_i) in zip(y, yerr)]
        ymax = [y_i + yerr_i for (y_i, yerr_i) in zip(y, yerr)]

    elif len(yerr) == 2:

        ymin, ymax = yerr

    # ax.plot(x, y)
    ax.fill_between(x, ymax, ymin, alpha=alpha_fill, color=color,
                    zorder=0)


fig = plt.figure()

row_content = df.train_batch_size
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

ax = fig.gca()
ax.set_rasterization_zorder(1)

for i, row_level in enumerate(row_levels):

    running_time_mean = []
    running_time_std = []

    for j, col_level in enumerate(col_levels):

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        running_time_mean.append(run_df['mean_running_time'].mean())
        running_time_std.append(run_df['mean_running_time'].std())

    ax.loglog()

    ax.set_ylim(0.01, 1)

    line, = ax.plot(col_levels,
                    running_time_mean,
                    label=r'$|\textbf{B}| = ' + str(row_level) + '$',
                    alpha=0.5,
                    zorder=0)

    errorfill(col_levels,
              running_time_mean,
              running_time_std,
              color=line.get_color(),
              alpha_fill=0.3, ax=ax)

    ax.set_xlabel(r'$I_B$')

    ax.set_ylabel(r'Running Time ($\mu \pm \sigma$)')

plt.legend(loc="upper right")

fig.set_size_inches(3.5, 2)

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.suptitle("Impact of Batch Replacement Interval $(I_B)$ on Mean Running Time")


fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_batch_interval_study_timing.eps',
            rasterized=True,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.0)
plt.show()
