import idx2numpy
import theano


def load():
  return (load_dataset("train"), load_dataset("t10k"))

def load_dataset(ds):
  x_path = "mnist/%s-images-idx3-ubyte" % ds
  y_path = "mnist/%s-labels-idx1-ubyte" % ds
  x = preprocess_xs(idx2numpy.convert_from_file(x_path))
  y = idx2numpy.convert_from_file(y_path)
  return (x, y)

def preprocess_xs(x):
  return x.reshape((-1, 28*28)).astype(theano.config.floatX) / 255.0

