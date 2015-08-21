#!/usr/bin/python
import argparse
import models
import mnist

parser = argparse.ArgumentParser()
parser.add_argument('network_structure', type=str)
parser.add_argument('h5_file', type=str)
args = parser.parse_args()

#TODO: Deal with duplication between this code and train_mnist.py
# Set up the network:
print "Setting up network..."

dimensions = [28*28] + [int(x) for x in args.network_structure.split("x")] + [10]
net = models.BatchTrainedModel(models.perceptron_model(dimensions))
net.read(args.h5_file)

print "Loading MNIST..."
(train_x, train_y), (test_x, test_y) = mnist.load()
print "Done Loading MNIST."

train_error = net.error_rate(train_x, train_y)
test_error = net.error_rate(test_x, test_y)
print "Train Error rate =", train_error
print "Test Error rate =", test_error
