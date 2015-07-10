import h5py
import numpy
import random
import theano
import theano.tensor as T

def shared_rand(dims, name, a=0.1):
  value = numpy.random.uniform(-a, a, dims).astype(theano.config.floatX)
  return theano.shared(value, name=name);

# Returnts <output neurons>
def add_linear_transform(inputs, dim_in, dim_out, name_prefix, add_weight):
  weights = shared_rand((dim_in, dim_out), name_prefix + ":weight")
  bias = shared_rand((dim_out,), name_prefix + ":bias")
  add_weight(weights)
  add_weight(bias)
  outputs = inputs.dot(weights) + bias
  return outputs

def gradient_updates(error, values, learning_rate):
  gs = T.grad(error, values)
  return [(v, v - learning_rate * g_v) for (v, g_v) in zip(values, gs)]

def write_shared_vars(fname, shared_vars):
  f = h5py.File(fname, "w")
  for s in shared_vars:
    v = s.get_value()
    d = f.create_dataset(s.name, v.shape, dtype=v.dtype)
    d[...] = v

def read_shared_vars(fname, shared_vars):
  f = h5py.File(fname, "w")
  for s in shared_vars:
    s.set_value(f[s.name])

#MultiLayer Perceptron
def perceptron_model(dims, activation=T.nnet.sigmoid, name_prefix=""):
  def model_builder(x, add_weight):
    layer = x
    for i in range(len(dims)-1):
      layer_prefix = name_prefix + (":%03d" % (i+1))
      layer = add_linear_transform(layer, dims[i], dims[i+1], layer_prefix, add_weight)
      if i == len(dims)-2:
        layer = T.nnet.softmax(layer)
      else:
        layer = activation(layer)
    return layer
  return model_builder

class BatchTrainedModel:
  def __init__(self, model_builder):
    self.weights = list()
    add_weight = lambda w: self.weights.append(w)
    self.x = T.matrix('x')
    self.y = T.ivector('y')

    self.p_y_given_x = model_builder(self.x, add_weight)
    self.pred = T.argmax(self.p_y_given_x, axis=1)
    self.error = -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
    self.predict_fn = theano.function([self.x], self.pred, name="predict")
    self.train_fn = theano.function([self.x, self.y], self.pred, name="train",
        updates=gradient_updates(self.error, self.weights, 0.01))

  def train_epoch(self, x_set, y_set, batch_size=32):
    order = range(len(x_set))
    random.shuffle(order)
    for i in xrange(0, len(order), batch_size):
      if i+batch_size <= len(order):
        subset = order[i:i+batch_size]
        self.train_fn(x_set[subset], y_set[subset])

  def error_rate(self, x_set, y_set):
    p = self.predict_fn(x_set)
    error = (p != y_set).sum()
    return error/float(x_set.shape[0])

  def write(self, fname):
    write_shared_vars(fname, self.weights)

  def read(self, fname):
    read_shared_vars(fname, self.weights)

