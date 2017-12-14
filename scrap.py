import itertools
import random


exp_design = [('rep_num', range(5)),
              ('train_batch_size', [128]),
              ('optimizer', ['csa_annealer',
                             'fsa_annealer',
                             'gsa_annealer',
                             'layerwise_csa_annealer',
                             'layerwise_fsa_annealer',
                             'layerwise_gsa_annealer']),
              ('init_temp', [5.0]),
              ('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
              ('batch_interval', [1, 10, 100, 1000, 10000])]

# Translate the design structure into flag strings.
exp_flag_strings = [['--' + f + '=' + str(v) for v in r]
                    for (f, r) in exp_design]

# Produce the Cartesian set of configurations.
experimental_configs = list(itertools.product(*exp_flag_strings))
experimental_configs = itertools.product(*exp_flag_strings)
# random.shuffle(experimental_configs)

# Iterate over each experimental configuration, launching a job for each.
for i, experimental_config in enumerate(experimental_configs):

  print(experimental_config)
