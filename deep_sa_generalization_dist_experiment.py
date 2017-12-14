

import os
import sys
import csv
import time
import random
import argparse
import itertools
import functools
import numpy as np
import tensorflow as tf
from shutil import copyfile

sys.path.append("../tensorflow-zoo")
from baseline_model import *

sys.path.append("../sapy")
from sapy import *
from temperature_functions import *

sys.path.append("../sapy/tensorflow")
from tfstate import *
from tfperturber import *
from tfevaluator import *
from tfacceptance import *


def deep_sa_experiment():

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)
    # # Unpack the experimental parameters.
    # (optimizer, batch_size, rep) = exp_parameters

    # Reset the default graph.
    tf.reset_default_graph()

    # Declare experimental measurement vars.
    steps = []
    val_losses = []
    val_errors = []
    train_losses = []
    train_errors = []
    mean_running_times = []

    print("------------model_output-------------")
    # Instantiate a model.
    model = Model(FLAGS.input_size,
                  FLAGS.label_size,
                  FLAGS.learning_rate,
                  FLAGS.train_enqueue_threads,
                  FLAGS.val_enqueue_threads,
                  FLAGS.data_dir,
                  FLAGS.train_file,
                  FLAGS.validation_file)
    print("-------------------------------------")

    # Instantiate TensorFlow  annealing objects.
    tf_state = TensorFlowState()
    tf_cost_evaluator = TensorFlowCostEvaluator(model.loss)

    if FLAGS.optimizer == 'csa_annealer':

        tf_perturber = CSATensorFlowPerturber(FLAGS.learning_rate)

    elif FLAGS.optimizer == 'fsa_annealer':

        tf_perturber = FSATensorFlowPerturber(FLAGS.learning_rate)

    elif FLAGS.optimizer == 'gsa_annealer':

        tf_perturber = GSATensorFlowPerturber(FLAGS.learning_rate, FLAGS.gsa_q)

    elif FLAGS.optimizer == 'layerwise_csa_annealer':

        tf_perturber = LayerwiseCSATensorFlowPerturber(FLAGS.learning_rate)

    elif FLAGS.optimizer == 'layerwise_fsa_annealer':

        tf_perturber = LayerwiseFSATensorFlowPerturber(FLAGS.learning_rate)

    elif FLAGS.optimizer == 'layerwise_gsa_annealer':

        tf_perturber = LayerwiseGSATensorFlowPerturber(FLAGS.learning_rate, FLAGS.gsa_q)

    elif FLAGS.optimizer == 'sgd':

        tf_perturber = []

    else:

        print("That is not a valid optimizer.")
        exit()

    # Get input data.
    image_batch, label_batch = model.get_train_batch_ops(
        batch_size=FLAGS.train_batch_size)

    (val_image_batch, val_label_batch) = model.get_val_batch_ops(
        batch_size=FLAGS.val_batch_size)

    # Merge the summary.
    tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=600.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir +
        #                                      '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        tf_state.start(sess=sess)
        tf_perturber.start(sess=sess)
        tf_cost_evaluator.start(sess=sess)

        # Instantiate an Annealer that, when called, increments the annealing.
        annealer = Annealer(tf_state,
                            tf_perturber,
                            tf_cost_evaluator,
                            fsa_temperature,
                            fsa_acceptance_probability,
                            FLAGS.init_temp)

        # Declare timekeeping vars.
        running_times = [0]
        optimize_step_running_time = 0

        print("------------training_output-------------")
        # Print a line for debug.
        print('step | train_loss | train_error | val_loss |' +
              ' val_error | t | total_time')

        # Load the validation set batch into memory.
        val_images, val_labels = sess.run([val_image_batch, val_label_batch])

        # Make a dict to load the val batch onto the placeholders.
        val_dict = {model.stimulus_placeholder: val_images,
                    model.target_placeholder: val_labels,
                    model.keep_prob: 1.0}

        print(sess.run(model.error, feed_dict=val_dict))

        # Iterate until max steps.
        for i in range(FLAGS.max_steps):

            # Check for break.
            if sv.should_stop():
                break

            # If it is a batch refresh interval, refresh the batch.
            if((i % FLAGS.batch_interval == 0) or (i == 0)):

                # Update the batch.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

            # Make a dict to load the batch onto the placeholders.
            train_dict = {model.stimulus_placeholder: train_images,
                          model.target_placeholder: train_labels,
                          model.keep_prob: FLAGS.keep_prob}

            # If we have reached a testing interval, test.
            if (i % FLAGS.test_interval == 0):

                # Compute error over the training set.
                train_error = sess.run(model.error, feed_dict=train_dict)

                # Compute loss over the training set.
                train_loss = sess.run(model.loss, feed_dict=train_dict)

                # Compute error over the validation set.
                val_error = sess.run(model.error, feed_dict=val_dict)

                # Compute loss over the validation set.
                val_loss = sess.run(model.loss, feed_dict=val_dict)

                # Store the data we wish to manually report.
                steps.append(i)
                train_losses.append(train_loss)
                train_errors.append(train_error)
                val_losses.append(val_loss)
                val_errors.append(val_error)
                mean_running_times.append(np.mean(running_times))

                # Print relevant values.
                print('%d | %.6f | %.2f | %.6f | %.2f | %.6f | %.2f'
                      % (i,
                         train_loss,
                         train_error,
                         val_loss,
                         val_error,
                         np.mean(running_times),
                         np.sum(running_times)))

            # Hack the start time.
            start_time = time.time()

            if FLAGS.optimizer == 'csa_annealer':

                annealer(input_data=train_dict)

            elif FLAGS.optimizer == 'fsa_annealer':

                annealer(input_data=train_dict)

            elif FLAGS.optimizer == 'gsa_annealer':

                annealer(input_data=train_dict)

            elif FLAGS.optimizer == 'layerwise_csa_annealer':

                tv_count = len(tf.trainable_variables())

                perturb_params = random.choice(range(tv_count))

                annealer(perturb_params=perturb_params, input_data=train_dict)

            elif FLAGS.optimizer == 'layerwise_fsa_annealer':

                tv_count = len(tf.trainable_variables())

                perturb_params = random.choice(range(tv_count))

                annealer(perturb_params=perturb_params, input_data=train_dict)

            elif FLAGS.optimizer == 'layerwise_gsa_annealer':

                tv_count = len(tf.trainable_variables())

                perturb_params = random.choice(range(tv_count))

                annealer(perturb_params=perturb_params, input_data=train_dict)

            elif FLAGS.optimizer == 'sgd':

                sess.run(model.optimize, feed_dict=train_dict)

            else:

                print("That is not a valid optimizer.")
                exit()

            # train_writer.add_summary(summary, i)

            # Update timekeeping variables.
            optimize_step_running_time = time.time() - start_time
            running_times.append(optimize_step_running_time)

        print("----------------------------------------")
        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)

        # Iterate over the results vectors for each config.
        for (step, tl, te, vl, ve, mrt) in zip(steps,
                                               train_losses,
                                               train_errors,
                                               val_losses,
                                               val_errors,
                                               mean_running_times):

            # Write the data to a csv.
            csvwriter.writerow([step,
                                tl,
                                te,
                                vl,
                                ve,
                                mrt])


    print("---------------copy_file-----------------")
    copy(FLAGS.log_dir + '/' + FLAGS.log_filename,
         FLAGS.log_dir + '/../')

    print("-----------------------------------------")
    # TODO: Copy logfile up one directory.

    return([steps, train_losses, val_losses, mean_running_times])

# Experimental dimensions
# Could hybridize SA and SGD completely - perturb gradients
# Could hybridize with parameter - slowly switch from SGD to SA
# Could hybridize with step - instantly switch from SGD to SA
# Could hybridize with optimization paradigms - diff paradigms through training


def main(_):

    return(deep_sa_experiment())


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.

    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data/mnist',
                        help='Directory from which to pull data.')

    parser.add_argument('--log_dir', type=str,
                        default='../log/deep_sa_generalization/templog',
                        help='Summaries log directory.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensinoality of the output space.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--val_enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue val examples.')

    parser.add_argument('--reanneal_interval', type=int,
                        default=500000,
                        help='Batch size.')

    parser.add_argument('--hl_size', type=int,
                        default=128,
                        help='Directory with the training data.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--init_temp', type=float,
                        default=5.0,
                        help='Initial temperature for SA algorithm')

    parser.add_argument('--rep_num', type=int,
                        default=0,
                        help='Flag identifying the repitition number.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa_generalization_experiment.csv',
                        help='Summaries log directory.')

    parser.add_argument('--train_enqueue_threads', type=int,
                        default=128,
                        help='Number of threads to enqueue train examples.')

    parser.add_argument('--train_batch_size', type=int,
                        default=128,
                        help='Training set batch size.')

    parser.add_argument('--batch_interval', type=int,
                        default=500000,
                        help='Interval between training batch refresh.')

    parser.add_argument('--optimizer', type=str,
                        default='gsa_annealer',
                        help='Optimizer to use.')

    parser.add_argument('--gsa_q', type=float,
                        default=2.1,
                        help='Optimizer to use.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
