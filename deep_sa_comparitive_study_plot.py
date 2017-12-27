from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import Grid

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa/deep_sa.csv')
# df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa_generalization/deep_sa_experiment.csv')

# max_running_time = np.max(df.running_time)
# print(max_running_time)
# min_running_time = 0

# max_queue_size = np.max(df.queue_size)
# print(max_queue_size)
# min_queue_size = 0

# max_step_num = np.max(df.step_num)
# print(max_queue_size)
# min_step_num = np.min(df.step_num)

# subplot_x_str = str(len(df.thread_count.unique()))
# subplot_y_str = str(len(df.batch_size.unique()))


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()

    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = [y_i - yerr_i for (y_i, yerr_i) in zip(y, yerr)]
        ymax = [y_i + yerr_i for (y_i, yerr_i) in zip(y, yerr)]
    elif len(yerr) == 2:
        ymin, ymax = yerr
    # ax.plot(x, y)
    ax.fill_between(x, ymax, ymin, alpha=alpha_fill, color=color)


fig = plt.figure()

plot_num = 0

row_content = df.train_batch_size
row_levels = row_content.unique()


col_content = df.optimizer
col_levels = col_content.unique()

intraplot_content = df.learning_rate
intraplot_levels = intraplot_content.unique()


for i, row_level in enumerate(row_levels):

    # print(df.loc[df['thread_count'] == tc])

    for j, col_level in enumerate(col_levels):

        # Create scatter axis here.
        plot_num += 1

        ax = fig.add_subplot(len(row_levels),
                             len(col_levels),
                             plot_num)

        show_xlabel = len(row_levels) == (i + 1)

        show_ylabel = j == 0

        annotate_col = i == 0
        col_annotation = 'Optimizer: \n' + col_level

        annotate_row = j == 0
        row_annotation = 'Train Set Size: \n' + str(row_level)

        for k, intraplot_level in enumerate(intraplot_levels):

            # ax.set_xlim(0.00001, 10)
            # ax.set_ylim(0.001, 1)

            run_df = df.loc[(row_content == row_level) &
                            (col_content == col_level) &
                            (intraplot_content == intraplot_level)]

            # if plot_loss:

            train_loss_mean = run_df.groupby(['step_num'])['train_loss'].mean().tolist()
            train_loss_std = run_df.groupby(['step_num'])['train_loss'].std().tolist()

            val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
            val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

            ax.set_yscale("log", nonposx='clip')

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



            # train_loss_mean = train_loss_mean[:2]
            # train_loss_std = train_loss_std[:2]

            # val_loss_mean = val_loss_mean[:2]
            # val_loss_std = val_loss_std[:2] 

            # step = step[:2]

            line, = ax.plot(step,
                            val_loss_mean,
                            label='alpha=' + str(intraplot_level),
                            alpha=0.5)

            errorfill(step,
                      val_loss_mean,
                      val_loss_std,
                      color=line.get_color(),
                      alpha_fill=0.3, ax=ax)

            ax.plot(step,
                    train_loss_mean,
                    "--",
                    color=line.get_color(),
                    label='alpha=' + str(intraplot_level),
                    alpha=0.5)

            # errorfill(step,
            #           train_loss_mean,
            #           train_loss_std, color=None, alpha_fill=0.3, ax=ax)

        if show_xlabel:

            ax.set_xlabel('Training Step')

        else:

            ax.xaxis.set_ticklabels([])

        if show_ylabel:

            ax.set_ylabel('Validation Set Loss (mean, std)')

        else:

            ax.yaxis.set_ticklabels([])

        if annotate_col:
            pad = 10
            ax.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        if annotate_row:
            pad = -100
            ax.annotate(row_annotation, xy=(0, 0.55), xytext=(pad, 0),
                        rotation=90,
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

plt.legend()

plt.suptitle("Generalization and Training Loss of SGD and SA by Training Set Size")

plt.show()
