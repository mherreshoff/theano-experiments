#!/usr/local/bin/python
import h5py
import numpy
import random

import mlp
import mnist

MAX_EPOCHS = 100
DIMENSIONS = [28*28, 10]

# Set up the network:
print "Setting up network..."
net = mlp.MultiLayerPerceptron(DIMENSIONS)

print "Loading MNIST..."
(train_x, train_y), (test_x, test_y) = mnist.load()
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
  pred = net.train(train_x, train_y);

