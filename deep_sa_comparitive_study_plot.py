from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa/deep_sa_comparitive_study.csv')

df = df.sort_values(['optimizer', 'learning_rate'])


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

plot_num = 0

row_content =  df.learning_rate
row_levels = row_content.unique()

col_content = df.optimizer
col_levels = col_content.unique()

intraplot_content = df.learning_rate
intraplot_levels = intraplot_content.unique()


for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        # Create scatter axis here.
        plot_num += 1

        ax = fig.add_subplot(len(row_levels),
                             len(col_levels),
                             plot_num)

        ax.set_rasterization_zorder(1)

        show_xlabel = len(row_levels) == (i + 1)

        show_ylabel = j == 0

        annotate_col = i == 0
        col_annotation = r'$' + str(col_level) + '$'

        annotate_row = j == 0
        row_annotation = r'$alpha= ' + str(row_level) + '$'

        # ax.set_xlim(0.00001, 10)
        # ax.set_ylim(0.001, 1)

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        # if plot_loss:

        train_loss_mean = run_df.groupby(['step_num'])['train_loss'].mean().tolist()
        train_loss_std = run_df.groupby(['step_num'])['train_loss'].std().tolist()

        val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
        val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

        # ax.set_yscale("log", nonposx='clip')

        ax.set_ylim(0.01, 15)

        # if plot_error:

        # train_loss_mean = run_df.groupby(['step_num'])['train_error'].mean().tolist()
        # train_loss_std = run_df.groupby(['step_num'])['train_error'].std().tolist()

        # val_loss_mean = run_df.groupby(['step_num'])['val_error'].mean().tolist()
        # val_loss_std = run_df.groupby(['step_num'])['val_error'].std().tolist()

        # ax.set_ylim(0, 1)

        print(len(val_loss_mean))

        step = run_df['step_num']
        step = run_df.groupby(['step_num'])['step_num'].mean().tolist()
        print(len(step))

        line, = ax.plot(step,
                        val_loss_mean,
                        label=r'Validation Loss ($ \mu \pm \sigma$)',
                        alpha=0.5,
                        zorder=0)

        errorfill(step,
                  val_loss_mean,
                  val_loss_std,
                  color=line.get_color(),
                  alpha_fill=0.3, ax=ax)

        ax.plot(step,
                train_loss_mean,
                "--",
                color=line.get_color(),
                label=r'Training Loss ($ \mu $)',
                alpha=0.5,
                zorder=0)

        # errorfill(step,
        #           train_loss_mean,
        #           train_loss_std, color=None, alpha_fill=0.3, ax=ax)

        if show_xlabel:

            ax.set_xlabel('Training Step')

        else:

            ax.xaxis.set_ticklabels([])

        if show_ylabel:

            ax.set_ylabel('Loss')

        else:

            ax.yaxis.set_ticklabels([])

        if annotate_col:
            pad = 5
            ax.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        if annotate_row:
            pad = -25
            ax.annotate(row_annotation, xy=(0, 0.6), xytext=(pad, 0),
                        rotation=90,
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')


plt.grid(True,
         zorder=0)
plt.legend(bbox_to_anchor=(0.5, 0.0),
           loc="lower left",
           mode="expand",
           bbox_transform=fig.transFigure,
           borderaxespad=0,
           ncol=3)

plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.925))

fig.set_size_inches(7.1, 3.5)

plt.suptitle("Impact of Batch Replacement Interval $(I_B)$ on Validation and Training Set Loss")
fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
            rasterized=True,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.05)
plt.show()
