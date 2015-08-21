#!/usr/bin/python
import argparse
import numpy
import os
import sys

import models
import mnist

# Parse commandline:
parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('output_dir', type=str)
parser.add_argument('network_structure', type=str)
args = parser.parse_args()

os.makedirs(args.output_dir)

# Set up the network:
print "Setting up network..."

dimensions = [28*28] + [int(x) for x in args.network_structure.split("x")] + [10]
net = models.BatchTrainedModel(models.perceptron_model(dimensions))

print "Loading MNIST..."
(train_x, train_y), (test_x, test_y) = mnist.load()
print "Done Loading MNIST."

print "%d training examples" % train_x.shape[0]

print "Training..."
graph_f = open("%s/graph.tsv" % args.output_dir, "w")
for i in xrange(args.max_epochs+1):
  print "Ran for", i, "epochs"
  net.write("%s/epoch_%04d.hdf5" % (args.output_dir, i))
  train_error = net.error_rate(train_x, train_y)
  test_error = net.error_rate(test_x, test_y)
  graph_f.write("%d\t%f\t%f\n" % (i, train_error, test_error))
  graph_f.flush()
  print "Train Error rate =", train_error
  print "Test Error rate =", test_error
  pred = net.train_epoch(train_x, train_y, batch_size=args.batch_size);

