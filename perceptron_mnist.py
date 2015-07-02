#!/usr/local/bin/python
import h5py
import numpy
import mnist
import random
import theano
import theano.tensor as T

theano.config.floatX = 'float32'

MAX_TRAINING_STEPS = 1000000
DIMENSIONS = [28*28, 500, 500, 10]
BATCH_SIZE = 32
MAX_GRAPH_TRAIN_ERROR = 0.1

def shared_rand(name, a, dims):
  return theano.shared(
    numpy.random.uniform(-a, a, dims).astype(theano.config.floatX),
    name=name)

def gradient_updates(error, values, learning_rate):
  gs = T.grad(error, values)
  return [(v, v - learning_rate * g_v) for (v, g_v) in zip(values, gs)]

def write_shared_vars(fname, shared_vars):
  f = h5py.File(fname, "w")
  for s in shared_vars:
    v = s.get_value()
    d = f.create_dataset(s.name, v.shape, dtype=v.dtype)
    d[...] = v


class MultiLayerPerceptron:
  def __init__(self, dims):
    self.Ws = list()
    self.bs = list()
    self.x = T.fmatrix('x')
    self.y = T.ivector('y')
    hidden = self.x
    for i in range(len(dims)-1):
      W = shared_rand("W%d" % (i+1), 0.1, (dims[i], dims[i+1]))
      b = shared_rand("b%d" % (i+1), 0.1, dims[i+1])
      self.Ws.append(W)
      self.bs.append(b)
      q = hidden.dot(W)+b
      if i+1 == len(dims) - 1:
        self.p_y_given_x = T.nnet.softmax(q)
      else:
        hidden = T.nnet.sigmoid(q)
    self.pred = T.argmax(self.p_y_given_x, axis=1)
    self.error = -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
    self.predict_fn = theano.function([self.x], self.pred, name="predict")
    self.train_fn = theano.function([self.x, self.y], [self.pred],
      updates=gradient_updates(self.error, self.Ws + self.bs, 0.01))

  def train(self, x_set, y_set):
    return self.train_fn(x_set, y_set)

  def error_rate(self, x_set, y_set):
    p = self.predict_fn(x_set)
    error = (p != y_set).sum()
    return error/float(x_set.shape[0])

  def write(self, file_name):
    write_shared_vars(file_name, self.Ws + self.bs)

# Set up the network:
print "Setting up network..."
net = MultiLayerPerceptron(DIMENSIONS)

print "Loading MNIST..."
(train_x, train_y), (test_x, test_y) = mnist.load()

print "Done Loading MNIST."

print "%d training examples" % train_x.shape[0]

print "Training..."
graph_f = open("output/graph.txt", "w")
for i in xrange(MAX_TRAINING_STEPS+1):
  k = random.randint(0, train_x.shape[0]-1-BATCH_SIZE)
  pred = net.train(train_x[k:k+BATCH_SIZE], train_y[k:k+BATCH_SIZE])
  if i % 4000 == 0:
    print "Ran for", i, "mini-batches."
    net.write("output/iter_%08d.hdf5" % i)
    train_error = net.error_rate(train_x, train_y)
    test_error = net.error_rate(test_x, test_y)
    if train_error < MAX_GRAPH_TRAIN_ERROR:
      graph_f.write("%d\t%f\t%f\n" % (i, train_error, test_error))
      graph_f.flush()
    print "Train Error rate =", train_error
    print "Test Error rate =", test_error

