import idx2numpy

print "Loading MNIST..."
train_x = idx2numpy.convert_from_file("mnist/train-images-idx3-ubyte")
train_y = idx2numpy.convert_from_file("mnist/train-labels-idx1-ubyte")
test_x = idx2numpy.convert_from_file("mnist/t10k-images-idx3-ubyte")
test_y = idx2numpy.convert_from_file("mnist/t10k-labels-idx1-ubyte")
print "Done Loading MNIST."

print train_x.sum(axis=(1,2))
