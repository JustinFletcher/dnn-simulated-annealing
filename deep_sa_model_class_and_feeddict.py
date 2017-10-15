

import os
import sys
import time
import argparse
import functools
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

from tensorflow.examples.tutorials.mnist import input_data


sys.path.append("../sapy")
from sapy import *
from temperature_functions import *

sys.path.append("../sapy/tensorflow")
from tfstate import *
from tfperturber import *
from tfevaluator import *


# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# function to print the tensor shape.  useful for debugging
def print_tensor_shape(tensor, string):
    ''''
    input: tensor and string to describe it
    '''

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


class Model:

    def __init__(self, stimulus_placeholder, target_placeholder, keep_prob):

        self.stimulus_placeholder = stimulus_placeholder
        self.target_placeholder = target_placeholder
        self.learning_rate = FLAGS.learning_rate
        self.inference
        self.loss
        self.optimize
        self.error
        self.keep_prob = keep_prob

    def variable_summaries(self, var):
            """Attach a lot of summaries to a Tensor
            (for TensorBoard visualization)."""

            with tf.name_scope('summaries'):

                mean = tf.reduce_mean(var)

                tf.summary.scalar('mean', mean)

                with tf.name_scope('stddev'):

                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

                tf.summary.scalar('stddev', stddev)

                tf.summary.scalar('max', tf.reduce_max(var))

                tf.summary.scalar('min', tf.reduce_min(var))

                tf.summary.histogram('histogram', var)

            return()

    def weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def bias_variable(self, shape):

        initial = tf.constant(0.1, shape=shape)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def conv2d(self, x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def inference(self, input=None):
        '''
        input: tensor of input image. if none, uses instantiation input
        output: tensor of computed logits
        '''

        print_tensor_shape(self.stimulus_placeholder, 'images shape')
        print_tensor_shape(self.target_placeholder, 'label shape')

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        images_re = tf.reshape(self.stimulus_placeholder, [-1, 28, 28, 1])
        print_tensor_shape(images_re, 'reshaped images shape')

        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            h_pool1 = self.max_pool_2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            h_pool2 = self.max_pool_2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, FLAGS.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout

    @define_scope
    def loss(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_placeholder, logits=self.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        return(loss)

    @define_scope
    def optimize(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_placeholder, logits=self.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        # Minimize the loss by incrementally changing trainable variables.
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.target_placeholder, 1),
                                tf.argmax(self.inference, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # tf.summary.scalar('error', error)
        return(error)


def create_model():

    # Build placeholders for the input and desired response.
    stimulus_placeholder = tf.placeholder(tf.float32, [None, 784])
    target_placeholder = tf.placeholder(tf.int32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Instantiate a model.
    model = Model(stimulus_placeholder, target_placeholder, keep_prob)

    return(model)


def train():

    model = create_model()

    # Instantiate a TensorFlow state object to be annealed.
    tf_state = TensorFlowState()

    # Instantiate a TensorFlow state perturber.
    tf_perturber = TensorFlowPerturberFSA(FLAGS.learning_rate)

    # Instantiate a TensorFlow cost evaluator.
    tf_cost_evaluator = TensorFlowCostEvaluator(model.loss)

    def reject(t, d):

        return(False)

    def fsa_acceptance_probability(t, d):

        return(np.exp(-d / t) > np.random.rand())

    merged = tf.summary.merge_all()

    # Get input data.
    mnist = input_data.read_data_sets(FLAGS.data_dir + '/mnist/', one_hot=True)

    # init_op = [tf.global_variables_initializer()]

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=10.0)
    # sess = sv.managed_session()

    with sv.managed_session() as sess:

        # sess.run(init_op)

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             # sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        tf_state.start(sess=sess)
        tf_perturber.start(sess=sess)
        tf_cost_evaluator.start(sess=sess)

        # Instantiate an Annealer that, when called, increments the annealing.
        annealer = Annealer(tf_state, tf_perturber, tf_cost_evaluator,
                            fsa_temperature, fsa_acceptance_probability,
                            FLAGS.init_temp)

        total_time = 0
        i_delta = 0

        print('step | loss | error | t | total_time')

        for i in range(FLAGS.max_steps):

            i_start = time.time()

            if sv.should_stop():
                break

            # If we have reached a testing interval, test.
            if i % FLAGS.test_interval == 0:

                # Load the full dataset.
                images, labels = mnist.test.images, mnist.test.labels

                # Compute error over the test set.
                error = sess.run(model.error,
                                 {model.stimulus_placeholder: images,
                                  model.target_placeholder: labels,
                                  model.keep_prob: 1.0})

                # Compute error over the test set.
                loss = sess.run(model.loss,
                                {model.stimulus_placeholder: images,
                                 model.target_placeholder: labels,
                                 model.keep_prob: 1.0})

                print_tuple = (i, loss, error, i_delta, total_time)
                print('%d | %.6f | %.2f | %.6f | %.2f' % print_tuple)

            # Iterate, training the network.
            else:

                # Grabe a batch
                images, labels = mnist.train.next_batch(FLAGS.batch_size)

                # Train the model on the batch.
                # sess.run(model.optimize,
                #          {model.stimulus_placeholder: images,
                #           model.target_placeholder: labels,
                #           model.keep_prob: 0.5})
                annealer(input_data={model.stimulus_placeholder: images,
                                     model.target_placeholder: labels,
                                     model.keep_prob: 1.0})

                # train_writer.add_summary(summary, i)

            i_stop = time.time()
            i_delta = i_stop - i_start
            total_time = total_time + i_delta

        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()


def main(_):

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    train()

# Instrumentation: Loss function stability by batch size.


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')

    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data',
                        help='Directory for storing input data')

    parser.add_argument('--log_dir', type=str,
                        default='../log/tensorboard',
                        help='Summaries log directory')

    parser.add_argument('--batch_size', type=int,
                        default=2**16,
                        help='Batch size.')

    parser.add_argument('--train_dir', type=str,
                        default='../data',
                        help='Directory with the training data.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--init_temp', type=float,
                        default=1.0,
                        help='Initial temperature for SA algorithm')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
