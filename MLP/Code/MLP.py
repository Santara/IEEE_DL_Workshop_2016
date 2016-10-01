"""
Contains functions that define the deep learning neural
network model, performs inference, computes loss, and defines
the model updates during training
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_INPUT_FEATURES = 64   
NUM_TARGET_ATTRIBUTES = 10

def inference(inputs, fc1_units):
    """Build the Multi-Layer perceptron model up to where it may be used for inference.
    Args:
    inputs: Input placeholder.
    fc1_units = Number of units in the first fully connected hidden layer
    Returns:
    logits: Output tensor with the computed logits.
    """
        
    # FC 1
    with tf.variable_scope('h_FC1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([NUM_INPUT_FEATURES, fc1_units],
                                stddev=1.0 / math.sqrt(float(NUM_INPUT_FEATURES))), name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]), name='biases')
        
        h_FC1 = tf.nn.tanh(tf.matmul(inputs, weights) + biases, name=scope.name)
        
    
    # Linear
    with tf.variable_scope('output_linear') as scope:
        weights = tf.Variable(
            tf.truncated_normal([fc1_units, NUM_TARGET_ATTRIBUTES],
                                stddev=1.0 / math.sqrt(float(fc1_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_TARGET_ATTRIBUTES]), name='biases')
        logits = tf.matmul(h_FC1, weights) + biases
    
    
    return logits


def loss(logits, targets):
    """Calculates the loss from the logits and the targets.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_TARGET_ATTRIBUTES].
    targets: Targets tensor, float - [batch_size, NUM_TARGET_ATTRIBUTES].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_float(targets)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='xentropy')
    
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.scalar_summary('cross_entropy_loss', loss)
    
    return loss


# Define the Training OP
# --------------------

# In[8]:

def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Define the Evaluation OP
# ----------------------

def evaluation(logits, targets):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size, NUM_TARGET_ATTRIBUTES]
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))
    # Return the number of true entries.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
