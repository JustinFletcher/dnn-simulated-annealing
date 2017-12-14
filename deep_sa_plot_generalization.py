from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import Grid

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa_generalization_dist_experiment/deep_sa_generalization_dist_experiment.csv')
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
    ax.fill_between(x, ymax, ymin, alpha=alpha_fill)


fig = plt.figure()

plot_num = 0

row_content = df.batch_interval
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

        for k, intraplot_level in enumerate(intraplot_levels):

            # ax.set_xlim(0.00001, 10)
            ax.set_ylim(0, 14)
            ax.set_ylim(0.001, 1)

            run_df = df.loc[(row_content == row_level) &
                            (col_content == col_level) &
                            (intraplot_content == intraplot_level)]

            # print(run_df)
            # Create some mock data

            # mean_val_loss = run_df.groupby([''])['val_loss'].mean()

            train_loss_mean = run_df.groupby(['step_num'])['train_loss'].mean().tolist()
            train_loss_std = run_df.groupby(['step_num'])['train_loss'].std().tolist()

            val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
            val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

            train_loss_mean = run_df.groupby(['step_num'])['train_error'].mean().tolist()
            train_loss_std = run_df.groupby(['step_num'])['train_error'].std().tolist()

            val_loss_mean = run_df.groupby(['step_num'])['val_error'].mean().tolist()
            val_loss_std = run_df.groupby(['step_num'])['val_error'].std().tolist()

            print(len(val_loss_mean))

            step = run_df['step_num']
            step = run_df.groupby(['step_num'])['step_num'].mean().tolist()
            print(len(step))
            # print(step)

            # print(len(train_loss))
            # print(len(mean_val_loss))
            # show_xlabel = len(df.thread_count.unique()) == (i + 1)
            # show_label_1 = j == 0
            # show_label_2 = len(df.batch_size.unique()) == (j + 1)

            # annotate_col = i == 0
            # col_annotation = 'Batch Size = %d' % bs

            # annotate_row = j == 0
            # row_annotation = 'Thread \n Count = %d' % tc

            # Create axes
            # ax.loglog()
            # ax.scatter(train_loss, val_loss, label=opt)
            # ax.plot(step, train_loss, '--', label=opt)
            # ax.plot(step, val_loss, label=opt)
            # ax.plot(step,
            #         train_loss_mean,
            #         '--',
            #         label=opt + '_train_learningrate=' + str(learning_rate),
            #         alpha=0.5)

            ax.plot(step,
                    val_loss_mean,
                    label= 'val_learningrate=' + str(intraplot_level),
                    alpha=0.5)

            # errorfill(step,
            #           train_loss_mean,
            #           train_loss_std, color=None, alpha_fill=0.3, ax=ax)

            errorfill(step,
                      val_loss_mean,
                      val_loss_std, color=None, alpha_fill=0.3, ax=ax)

            # ax.set_yscale("log", nonposx='clip')

        ax.legend()


    pad = -70
    ax.annotate(str(row_level), xy=(0, 0.75), xytext=(pad, 0),
                rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
plt.suptitle("Generalization and Training Loss of SGD and SA by Training Set Size")

plt.show()
