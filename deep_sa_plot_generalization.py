from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import Grid

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/sa/sa_generalization_out.csv')

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


for i, bs in enumerate(df.batch_size.unique()):

    # print(df.loc[df['thread_count'] == tc])

    for j, hl in enumerate(df.hl_size.unique()):

        # Create scatter axis here.
        plot_num += 1

        ax = fig.add_subplot(len(df.batch_size.unique()),
                             len(df.hl_size.unique()),
                             plot_num)

        ax.set_xlim(0.001, 10)
        ax.set_ylim(0.1, 1000)

        for k, opt in enumerate(df.optimizer.unique()):

            run_df = df.loc[(df['batch_size'] == bs) &
                            (df['hl_size'] == hl) &
                            (df['optimizer'] == opt)]

            # print(run_df)
            # Create some mock data

            # mean_val_loss = run_df.groupby([''])['val_loss'].mean()

            train_loss = run_df['train_loss']
            val_loss = run_df['val_loss']

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
            ax.scatter(train_loss, val_loss)

        pad = -70
        ax.annotate(str(bs), xy=(0, 0.75), xytext=(pad, 0),
                    rotation=90,
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

plt.suptitle("TensorFlow Queue Exhaustion on Hokule'a")

plt.show()
