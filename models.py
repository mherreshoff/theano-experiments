from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

#MultiLayer Perceptron
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
    self.model.fit(x_set, np_utils.to_categorical(y_set),
        nb_epoch=1, batch_size=32, verbose=1)

  def error_rate(self, x_set, y_set):
    p = self.model.predict_classes(x_set)
    error = (p != y_set).sum()
    return error/float(x_set.shape[0])

  def write(self, file_name):
    self.model.save_weights(file_name)


#Convolution + MultiLayerPerceptron
class ConvolutionalNet:
#TODO: add back arguments.
  def __init__(self):
    self.model = Sequential()
    self.model.add(Convolution2D(10, 1, 3, 3, activation='relu'))
    self.model.add(MaxPooling2D(poolsize=(4,4)))
    self.model.add(Flatten())
    self.model.add(Dense(6*6*10, 10, init='uniform', activation='softmax'))
    self.sgd = SGD(lr=0.001)
    self.model.compile(optimizer=self.sgd, loss='categorical_crossentropy')

  def train(self, x_set, y_set):
    self.model.fit(x_set.reshape((-1, 1, 28, 28)),
                   np_utils.to_categorical(y_set),
                   nb_epoch=1, batch_size=32, verbose=1)

  def error_rate(self, x_set, y_set):
    p = self.model.predict_classes(x_set.reshape((-1, 1, 28, 28)))
    error = (p != y_set).sum()
    return error/float(x_set.shape[0])

  def write(self, file_name):
    self.model.save_weights(file_name)

