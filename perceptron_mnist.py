import idx2numpy
import numpy
import random
import theano
import theano.tensor as T

training_steps = 1000000
eval_samples = 2000

# Set up the network:
print "Setting up network..."

# Add a left-most column to the matrix containing all ones.
def ones_prefix(matrix):
  return T.concatenate([T.ones((matrix.shape[0], 1)), matrix], axis=1)

def shared_zeros(dims):
  return theano.shared(numpy.zeros(dims).astype('float32'))

def shared_rand(a, dims):
  return theano.shared(numpy.random.uniform(-a, a, dims).astype('float32'))

def gradient_updates(error, values, learning_rate):
  gs = T.grad(error, values)
  return [(v, v - learning_rate * g_v) for (v, g_v) in zip(values, gs)]

theano.config.floatX = 'float32'
INPUT_DIM = 28*28
HIDDEN_DIM = 100
OUTPUT_DIM = 10
x = T.imatrix('x')
y = T.ivector('y')
x_n = ones_prefix(T.cast(x, 'float32') / 255.0)

W1 = shared_rand(0.1, (1+INPUT_DIM, HIDDEN_DIM))
W2 = shared_rand(0.1, (1+HIDDEN_DIM, OUTPUT_DIM))
# W = shared_zeros((1+INPUT_DIM, OUTPUT_DIM))

hidden = ones_prefix(T.nnet.sigmoid(x_n.dot(W1)))
p_y_given_x = T.nnet.softmax(hidden.dot(W2))
pred = T.argmax(p_y_given_x, axis=1)
error = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

predict = theano.function([x], pred, name="predict")
train = theano.function([x, y], [pred],
  updates=gradient_updates(error, [W1, W2], 0.01))

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
    print "Ran for", i, "mini-batches."
    print "Train Error rate =", error_rate(train_x, train_y)
    print "Test Error rate =", error_rate(test_x, test_y)

