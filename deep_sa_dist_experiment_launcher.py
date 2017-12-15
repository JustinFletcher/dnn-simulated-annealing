
#!/usr/bin/python
# Example PBS cluster job submission in Python

import csv
import time
import random
import argparse
import itertools
import subprocess
import tensorflow as tf

# If you want to be emailed by the system, include these in job_string:
#PBS -M your_email@address
#PBS -m abe  # (a = abort, b = begin, e = end)


def main(FLAGS):

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # # Declare experimental flags.
    exp_design = [('rep_num', range(5)),
                  ('train_batch_size', [128, 2048, 8192]),
                  ('optimizer', ['csa_annealer',
                                 'fsa_annealer',
                                 'gsa_annealer',
                                 'layerwise_csa_annealer',
                                 'layerwise_fsa_annealer',
                                 'layerwise_gsa_annealer']),
                  ('init_temp', [5.0]),
                  ('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
                  ('batch_interval', [1000000])]

    # exp_design = [('rep_num', range(2)),
    #               ('train_batch_size', [128]),
    #               ('optimizer', ['csa_annealer',
    #                              'fsa_annealer',
    #                              'gsa_annealer',
    #                              'layerwise_csa_annealer',
    #                              'layerwise_fsa_annealer',
    #                              'layerwise_gsa_annealer']),
    #               ('init_temp', [5.0]),
    #               ('learning_rate', [0.001]),
    #               ('batch_interval', [1])]

    # Translate the design structure into flag strings.
    exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                        for (f, r) in exp_design]

    # Produce the Cartesian set of configurations.
    experimental_configs = list(itertools.product(*exp_flag_strings))

    # Shuffle the submission order of configs to avoid asymetries.
    random.shuffle(experimental_configs)

    # Make a list to store job id strings.
    job_ids = []
    input_output_maps = []

    # Iterate over each experimental configuration, launching a job for each.
    for i, experimental_config in enumerate(experimental_configs):

        print("-----------experimental_config---------")
        print(experimental_config)
        print("---------------------------------------")

        # Use subproces to command qsub to submit a job.
        p = subprocess.Popen('qsub',
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             shell=True)

        # Customize your options here.
        job_name = "dist_ex_%d" % i
        walltime = "4:00:00"
        select = "1:ncpus=20:mpiprocs=20"
        command = "python " + FLAGS.experiment_py_file

        # Iterate over flag strings, building the command.
        for flag in experimental_config:

            command += ' ' + flag

        # Add a final flag modifying the log filename to be unique.
        log_filename = 'templog' + str(i)

        temp_log_dir = FLAGS.log_dir + 'templog' + str(i) + '/'

        command += ' --log_dir=' + temp_log_dir

        # Add the logfile to the command.
        command += ' --log_filename=' + log_filename

        # log_filenames.append(log_filename)

        # Build IO maps.
        input_output_map = (experimental_config, temp_log_dir, log_filename)
        input_output_maps.append(input_output_map)

        # Build the job string.
        job_string = """#!/bin/bash
        #PBS -N %s
        #PBS -l walltime=%s
        #PBS -l select=%s
        #PBS -o ~/log/output/%s.out
        #PBS -e ~/log/error/%s.err
        #PBS -A MHPCC96670DA1
        #PBS -q standard
        module load anaconda2
        module load tensorflow/1.0.0
        module load cudnn/5.1
        cd $PBS_O_WORKDIR
        %s""" % (job_name, walltime, select, job_name, job_name, command)

        # Print your job string.
        print(job_string)

        # Send job_string to qsub.
        job_ids.append(p.communicate(job_string)[0])
        time.sleep(1)

        print("-----------------")

    jobs_complete = False
    timeout = False
    elapsed_time = 0

    # Loop until timeout or all jobs complete.
    while not(jobs_complete) and not(timeout):

        print("-----------------")

        print('Time elapsed: ' + str(elapsed_time) + ' seconds.')

        time.sleep(10)

        elapsed_time += 10

        # Create a list to hold the Bool job complete flags
        job_complete_flags = []

        # Iterate over each job id string.
        for job_id in job_ids:

            # TODO: Handle job completion gracefully.

            # Issue qstat command to get job status.
            p = subprocess.Popen('qstat -r ' + job_id,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 shell=True)

            output = p.communicate()

            try:
                # Read the qstat stdout, parse the state, and conv to Boolean.
                job_complete = output[0].split()[-2] == 'E'

            except:

                job_complete = True

            # Print a diagnostic.
            print('Job ' + job_id[:-1] + ' complete? ' +
                  str(job_complete) + '.')

            job_complete_flags.append(job_complete)

            if job_complete:

                p = subprocess.Popen('qdel -Wforce ' + job_id,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     shell=True)

        # And the job complete flags together.
        jobs_complete = (all(job_complete_flags))

        # Check if we've reached timeout.
        timeout = (elapsed_time > FLAGS.max_runtime)

        # # Accomodate Python 3+
        # with open(FLAGS.log_dir '/' + FLAGS.log_filename, 'w') as csvfile:

        # Accomodate Python 2.7 on Hokulea.
        with open(FLAGS.log_dir + '/' + FLAGS.log_filename, 'wb') as csvfile:

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
            for (input_flags,
                 output_dir,
                 output_filename) in input_output_maps:

                input_row = []

                # Process the flags into output values.
                for flag in input_flags:

                    flag_val = flag.split('=')[1]

                    input_row.append(flag_val)

                output_file = output_dir + output_filename

                # Check if the output file has been written.
                if tf.gfile.Exists(output_file):

                    print("-----------remove_model_ckpt------------")

                    # tf.gfile.Remove(output_dir + "model.ckpt*")

                    with open(output_file, 'rb') as f:

                        reader = csv.reader(f)

                        for output_row in reader:

                            csvwriter.writerow(input_row + output_row)

                    print("---------------------------------------")

                else:

                    print("output filename not found: " + output_filename)

        print("-----------------")

    print("All jobs complete. Exiting.")


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='../log/deep_sa/',
                        help='Summaries log directory.')

    parser.add_argument('--log_filename', type=str,
                        default='deep_sa.csv',
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
