from keras.models import Sequential
from keras.layers.core import Dense
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

