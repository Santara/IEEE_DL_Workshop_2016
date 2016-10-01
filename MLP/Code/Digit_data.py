
""" Functions for handling the Tata Steel data"""
import os
import numpy
import pickle as pkl
import collections

class DataSet(object):

  def __init__(self,
               inputs,
               targets):
    """Construct a DataSet.
    """
    
    assert inputs.shape[0] == targets.shape[0], (
        'inputs.shape: %s targets.shape: %s' % (inputs.shape, targets.shape))
    
    self._num_examples = inputs.shape[0]    
    self._inputs = inputs
    self._targets = targets
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs

  @property
  def targets(self):
    return self._targets

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._inputs = self._inputs[perm]
      self._targets = self._targets[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._inputs[start:end], self._targets[start:end]

def read_data_sets(train_dir, file_name, training_percentage=70.0):    
    Datasets = collections.namedtuple('Datasets',['train', 'validation', 'test'])
    data_file = os.path.join(train_dir,file_name) 
    with open(data_file,'rb') as f:
        data = pkl.load(f)

    inputs = data['X']
    targets = data['y_']
    
    TRAINING_DATA_SIZE = int(len(inputs)*(training_percentage/100.0))
    VALIDATION_DATA_SIZE = int((len(inputs) - TRAINING_DATA_SIZE)/2)
    
    train_inputs = inputs[:TRAINING_DATA_SIZE]
    train_targets = targets[:TRAINING_DATA_SIZE]
    
    validation_inputs = inputs[TRAINING_DATA_SIZE: TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE]
    validation_targets = targets[TRAINING_DATA_SIZE: TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE]
    
    test_inputs = inputs[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE: ]
    test_targets = targets[TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE: ]
        
    
    train = DataSet(train_inputs, train_targets)
    validation = DataSet(validation_inputs, validation_targets)
    test = DataSet(test_inputs, test_targets)

    return Datasets(train=train, validation=validation, test=test)

def load_digit_data(training_percentage=70.0):
    return read_data_sets('../Data', 'digit_data.pkl',training_percentage)
