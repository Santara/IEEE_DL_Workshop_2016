from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from six.moves import xrange  
import tensorflow as tf

import MLP as model
import Digit_data as input_data


LEARNING_RATE = 0.01
MAX_STEPS = 10000
NUM_HIDDEN_1 = 50
BATCH_SIZE = 10
TRAIN_DIR = '../Data'


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    targets_placeholder: Targets placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and target tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, model.NUM_INPUT_FEATURES))
    targets_placeholder = tf.placeholder(tf.float32, shape=(batch_size, model.NUM_TARGET_ATTRIBUTES))
    return inputs_placeholder, targets_placeholder


def fill_feed_dict(data_set, inputs_pl, targets_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    inputs_pl: The images placeholder, from placeholder_inputs().
    targets_pl: The labels placeholder, from placeholder_inputs().
    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    inputs_feed, targets_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
      inputs_pl: inputs_feed,
      targets_pl: targets_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_accuracy,
            inputs_placeholder,
            targets_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_accuracy: The Tensor that returns the accuracy.
    inputs_placeholder: The images placeholder.
    targets_placeholder: The targets placeholder.
    data_set: The set of inputs and targets to evaluate, from
      input_data.read_data_sets().
    """

    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    accuracy = 0
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   inputs_placeholder,
                                   targets_placeholder)
        accuracy += sess.run(eval_accuracy, feed_dict=feed_dict)
    
    accuracy = accuracy / steps_per_epoch
    print('  Num examples: %d  accuracy: %0.04f' %
        (num_examples, accuracy))



def run_training():

    """Train model for a number of steps"""
    
    data_sets = input_data.read_data_sets(TRAIN_DIR,'digit_data.pkl')
    Training_data = data_sets.train
    Validation_data = data_sets.validation
    Test_data = data_sets.test
        
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
        inputs_placeholder, targets_placeholder = placeholder_inputs(BATCH_SIZE)

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(inputs_placeholder, NUM_HIDDEN_1)

        # Add to the Graph the Ops for loss calculation.
        loss = model.loss(logits, targets_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.training(loss, LEARNING_RATE)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_accuracy = model.evaluation(logits, targets_placeholder)

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(MAX_STEPS):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(Training_data,
                                     inputs_placeholder,
                                     targets_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, logit, accuracy =sess.run([train_op, loss, logits, eval_accuracy],
                                   feed_dict=feed_dict) 

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
            # Print status to stdout.
                print('Step %d: loss = %.2f, accuracy = %.2f (%.3f sec)' % (step, loss_value, accuracy, duration))

            # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_accuracy,
                        inputs_placeholder,
                        targets_placeholder,
                        Training_data)
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_accuracy,
                        inputs_placeholder,
                        targets_placeholder,
                        Validation_data)
                print('Test Data Eval:')
                do_eval(sess,
                        eval_accuracy,
                        inputs_placeholder,
                        targets_placeholder,
                        Test_data)
if __name__=='__main__':
	run_training()
