#!/usr/local/bin/python
import h5py
import idx2numpy
import numpy
import random
import theano
import theano.tensor as T

training_steps = 1000000
eval_samples = 2000

# Set up the network:
print "Setting up network..."

def shared_rand(name, a, dims):
  return theano.shared(
    numpy.random.uniform(-a, a, dims).astype('float32'),
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

theano.config.floatX = 'float32'
INPUT_DIM = 28*28
HIDDEN_DIM = 100
OUTPUT_DIM = 10
x = T.imatrix('x')
y = T.ivector('y')
x_n = (T.cast(x, 'float32') / 255.0)

W1 = shared_rand("W1", 0.1, (INPUT_DIM, HIDDEN_DIM))
b1 = shared_rand("b1", 0.1, HIDDEN_DIM)
W2 = shared_rand("W2", 0.1, (HIDDEN_DIM, OUTPUT_DIM))
b2 = shared_rand("b2", 0.1, OUTPUT_DIM)

hidden = T.nnet.sigmoid(x_n.dot(W1)+b1)
p_y_given_x = T.nnet.softmax(hidden.dot(W2)+b2)
pred = T.argmax(p_y_given_x, axis=1)
error = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

predict = theano.function([x], pred, name="predict")
train = theano.function([x, y], [pred],
  updates=gradient_updates(error, [W1, b1, W2, b2], 0.01))

def error_rate(x_set, y_set):
  p = predict(x_set)
  error = (predict(x_set) != y_set).sum()
  return error/float(x_set.shape[0])

print "Loading MNIST..."
train_x = idx2numpy.convert_from_file("mnist/train-images-idx3-ubyte")
train_x = train_x.reshape((-1, 28*28))
train_y = idx2numpy.convert_from_file("mnist/train-labels-idx1-ubyte")
test_x = idx2numpy.convert_from_file("mnist/t10k-images-idx3-ubyte")
test_x = test_x.reshape((-1, 28*28))
test_y = idx2numpy.convert_from_file("mnist/t10k-labels-idx1-ubyte")
print "Done Loading MNIST."

print "Training..."
for i in range(training_steps):
  k = random.randint(0, train_x.shape[0]-1-1000)
  pred = train(train_x[k:k+1000], train_y[k:k+1000])
  if i % 200 == 0:
    write_shared_vars("output/iter_%05d.hdf5" % i, [W1, W2])
    print "Ran for", i, "mini-batches."
    print "Train Error rate =", error_rate(train_x, train_y)
    print "Test Error rate =", error_rate(test_x, test_y)

