from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import Grid

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa_generalization_dist_experiment/deep_sa_generalization_dist_experiment.csv')
# 
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


fig = plt.figure()

plot_num = 0


for i, bs in enumerate(df.train_batch_size.unique()):

    # print(df.loc[df['thread_count'] == tc])



    for j, opt in enumerate(df.optimizer.unique()):

        # Create scatter axis here.
        plot_num += 1

        ax = fig.add_subplot(len(df.train_batch_size.unique()),
                             len(df.optimizer.unique()),
                             plot_num)


        # ax.set_xlim(0.00001, 10)
        ax.set_ylim(1e-7, 10)

        for l, init_temp in enumerate(df.init_temp.unique()):

            run_df = df.loc[(df['train_batch_size'] == bs) &
                            (df['optimizer'] == opt) &
                            (df['init_temp'] == init_temp)]

            # print(run_df)
            # Create some mock data

            # mean_val_loss = run_df.groupby([''])['val_loss'].mean()

            train_loss = run_df['train_loss']
            train_loss = run_df.groupby(['step_num'])['train_loss'].mean().tolist()
            # val_loss = run_df['val_loss']
            # print(val_loss)
            val_loss = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
            print(len(val_loss))
            # val_loss = (run_df['val_loss'].groupby(run_df['step_num']).mean())
            # print(val_loss)

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
            ax.loglog()
            # ax.scatter(train_loss, val_loss, label=opt)
            ax.plot(step, train_loss, '--', label=opt)
            ax.plot(step, val_loss, label=opt)

            ax.set_yscale("log", nonposx='clip')

            ax.legend()


    pad = -70
    ax.annotate(str(bs), xy=(0, 0.75), xytext=(pad, 0),
                rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
plt.suptitle("Generalization and Training Loss of SGD and SA by Training Set Size")

plt.show()
