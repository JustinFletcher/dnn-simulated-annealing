from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/Justi/Research/log/deep_sa/deep_sa_gsa_study.csv')

df = df.sort_values(['train_batch_size', 'batch_interval'])

fig = plt.figure()

plot_num = 0

row_content = df.train_batch_size
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()


# "batch_size", "learning_rate", "train_loss_mean", "train_loss_std", "val_loss_mean", "val_loss_std", "train_error_mean", "train_error_std", "val_error_mean", "val_error_std", "running_time_mean", "running_time_std"

# print("| batch_size | learning_rate || train_loss_mean (train_loss_std) | val_loss_mean (val_loss_std) | train_error_mean (train_error_std) | val_error_mean (val_error_std) | running_time_mean (running_time_std)")

print("\\toprule")


for i, row_level in enumerate(row_levels):

    # table_string = "\\ multirow{" + len(col_levels) + "}{*}{" + str(row_level) + "} & X \\"

    for j, col_level in enumerate(col_levels):

        # table_string += "& " + str(col_level)

        # Create scatter axis here.
        plot_num += 1

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]


        train_loss_mean = run_df.groupby(['step_num'])['train_loss'].mean().tolist()[-1]
        train_loss_std = run_df.groupby(['step_num'])['train_loss'].std().tolist()[-1]

        val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()[-1]
        val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()[-1]

        train_error_mean = run_df.groupby(['step_num'])['train_error'].mean().tolist()[-1]
        train_error_std = run_df.groupby(['step_num'])['train_error'].std().tolist()[-1]

        val_error_mean = run_df.groupby(['step_num'])['val_error'].mean().tolist()[-1]
        val_error_std = run_df.groupby(['step_num'])['val_error'].std().tolist()[-1]

        running_time_mean = run_df['mean_running_time'].mean()
        running_time_std = run_df['mean_running_time'].std()

        row_list = [row_level,
                    col_level,

                    train_loss_mean,
                    train_loss_std,

                    val_loss_mean,
                    val_loss_std,

                    train_error_mean,
                    train_error_std,

                    val_error_mean,
                    val_error_std,

                    running_time_mean,
                    running_time_std]

        # print(row_list)

        # table_string += "\\" + "\n"

        if j == 0:

            row_label = "\multirow{ " + str(len(col_levels)) + "}{*}{" + str(row_level) + "}"

        else:

            row_label = ""

        row_indent1_label = col_level

        print('     %s &     %d &  %5.3f $\pm$ %5.3f & %5.3f $\pm$ %5.3f & %5.3f $\pm$ %5.3f & %5.3f $\pm$ %5.3f &  %5.3f $\pm$ %5.3f \\\\ ' % (row_label,
                    row_indent1_label,
                    train_loss_mean,
                    train_loss_std,
                    val_loss_mean,
                    val_loss_std,
                    train_error_mean,
                    train_error_std,
                    val_error_mean,
                    val_error_std,
                    running_time_mean,
                    running_time_std))

        if j == len(col_levels) - 1:

            if i == len(row_levels) - 1:

                print("\\bottomrule")

            else:

                print("\cmidrule(r){2-" + str(7) + "}")