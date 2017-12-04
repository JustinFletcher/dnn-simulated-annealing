
#!/usr/bin/python
# Example PBS cluster job submission in Python

import csv
import time
import argparse
import itertools
import subprocess
import tensorflow as tf

# If you want to be emailed by the system, include these in job_string:
#PBS -M your_email@address
#PBS -m abe  # (a = abort, b = begin, e = end)


def main(FLAGS):

    # Declare experimental flags.
    exp_design = [('rep_num', range(10)),
                  ('train_batch_size', [128, 2048, 16384]),
                  ('optimizer', ['sgd',
                                 'fsa_annealer',
                                 'layerwise_fsa_annealer']),
                  ('init_temp', [1.0, 6.0])]

    # Translate the design structure into flag strings.
    exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                        for (f, r) in exp_design]

    # Produce the Cartesian set of configurations.
    experimental_configs = itertools.product(*exp_flag_strings)

    # Make a list to store job id strings.
    input_output_maps = []

    # Iterate over each experimental configuration, launching a job for each.
    for i, experimental_config in enumerate(experimental_configs):

        print("-----------experimental_config---------")
        print(experimental_config)
        print("---------------------------------------")

        # Add a final flag modifying the log filename to be unique.
        log_filename = 'templog' + str(i)

        # Build IO maps.
        input_output_map = (experimental_config, log_filename)
        input_output_maps.append(input_output_map)

        print("-----------------")

    # Accomodate Python 3+
    with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'w') as csvfile:

    # # Accomodate Python 2.7 on Hokulea.
    # with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

        # Parse out experiment parameter headers.
        parameter_labels = [flag_string for (flag_string, _) in exp_design]

        # Manually note response varaibles (MUST: Couple with experiment).
        response_labels = ['step_num',
                           'train_loss',
                           'train_error',
                           'val_loss',
                           'val_error',
                           'mean_running_time']
        # Join lists.
        headers = parameter_labels + response_labels

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Iterate over each eperimental mapping and write out.
        for (input_flags, output_filename) in input_output_maps:

            input_row = []

            # Process the flags into output values.
            for flag in input_flags:

                flag_val = flag.split('=')[1]

                input_row.append(flag_val)

            try:

                with open(FLAGS.log_dir + '/' + output_filename, 'r') as f:

                    reader = csv.reader(f)

                    for output_row in reader:

                        csvwriter.writerow(input_row + output_row)
            except:

                print("output filename not found: " + output_filename)


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='C:/Users/Justi/Research/log/deep_sa_generalization_dist_experiment',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa_generalization_dist_experiment.csv',
                        help='Merged output filename.')

    parser.add_argument('--max_runtime', type=int,
                        default=3600000,
                        help='Number of seconds to run before giving up.')

    parser.add_argument('--experiment_py_file', type=str,
                        default='~/dnn-simulated-annealing/deep_sa_generalization_dist_experiment.py',
                        help='Number of seconds to run before giving up.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
