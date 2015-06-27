import idx2numpy
import numpy
import random
import theano
import theano.tensor as T

print "Loading MNIST..."
train_x = idx2numpy.convert_from_file("mnist/train-images-idx3-ubyte")
train_x = train_x.reshape((-1, 28*28))
train_y = idx2numpy.convert_from_file("mnist/train-labels-idx1-ubyte")
test_x = idx2numpy.convert_from_file("mnist/t10k-images-idx3-ubyte")
test_x = test_x.reshape((-1, 28*28))
test_y = idx2numpy.convert_from_file("mnist/t10k-labels-idx1-ubyte")
print "Done Loading MNIST."

training_steps = 1000000
eval_samples = 2000

# Set up the network:

theano.config.floatX = 'float32'
INPUT_DIM = 28*28
OUTPUT_DIM = 10
x = T.imatrix('x')
y = T.ivector('y')
x_vec = T.cast(x, 'float32') / 255.0
W = theano.shared(numpy.zeros((INPUT_DIM, OUTPUT_DIM)).astype('float32'))
b = theano.shared(numpy.zeros(OUTPUT_DIM).astype('float32'))
p_y_given_x = T.nnet.softmax(T.dot(x_vec, W)+b)
pred = T.argmax(p_y_given_x, axis=1)
error = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

g_W, g_b = T.grad(error, [W, b])

predict = theano.function([x], pred, name="predict")
train = theano.function([x, y], [pred],
    updates=[(W, W - 0.01 * g_W), (b, b - 0.01 * g_b)])

def error_rate(x_set, y_set):
  p = predict(x_set)
#  print "p=",p
#  print "y_set=",y_set
  error = (predict(x_set) != y_set).sum()
  return error/float(x_set.shape[0])

for i in range(training_steps):
  k = random.randint(0, train_x.shape[0]-1-1000)
  pred = train(train_x[k:k+1000], train_y[k:k+1000])
  print "Ran for ", i, "steps."
  print "Train Error rate=", error_rate(train_x, train_y)
  print "Test Error rate=", error_rate(test_x, test_y)

