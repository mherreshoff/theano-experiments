#!/usr/local/bin/python
import h5py
import numpy
import random
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

import mnist

MAX_EPOCHS = 100
DIMENSIONS = [28*28, 10]
BATCH_SIZE = 32
MAX_GRAPH_TRAIN_ERROR = 0.1

class MultiLayerPerceptron:
  def __init__(self, dims):
    self.model = Sequential()
    for i in range(len(dims)-1):
      if i+1 == len(dims) - 1:
        activation = "softmax"
      else:
        activation = "sigmoid"
      self.model.add(Dense(dims[i], dims[i+1], init='uniform', activation=activation))
    self.model.compile(optimizer='sgd', loss='categorical_crossentropy')

  def train(self, x_set, y_set):
    self.model.fit(x_set, y_set, nb_epoch=1, batch_size=BATCH_SIZE, verbose=1)

  def error_rate(self, x_set, y_set):
    p = self.model.predict_classes(x_set)
    error = (p != y_set).sum()
    return error/float(x_set.shape[0])

  def write(self, file_name):
    self.model.save_weights(file_name)

# Set up the network:
print "Setting up network..."
net = MultiLayerPerceptron(DIMENSIONS)

print "Loading MNIST..."
(train_x, train_y), (test_x, test_y) = mnist.load()
train_y_class = np_utils.to_categorical(train_y, 10)
test_y_class = np_utils.to_categorical(train_y, 10)
print "Done Loading MNIST."

print "%d training examples" % train_x.shape[0]

print "Training..."
graph_f = open("output/graph.txt", "w")
for i in xrange(MAX_EPOCHS+1):
  print "Ran for", i, "epochs"
  net.write("output/epoch_%04d.hdf5" % i)
  train_error = net.error_rate(train_x, train_y)
  test_error = net.error_rate(test_x, test_y)
  graph_f.write("%d\t%f\t%f\n" % (i, train_error, test_error))
  graph_f.flush()
  print "Train Error rate =", train_error
  print "Test Error rate =", test_error
  pred = net.train(train_x, train_y_class);

