#!/usr/bin/python

from sklearn.decomposition import PCA
import numpy as np
import h5py

import mnist
import draw

def one_hot(vec, n_classes):
  r = np.zeros((len(vec), n_classes))
  for i, x in enumerate(vec):
    r[i][x] = 1
  return r

(train_x, train_y), (test_x, test_y) = mnist.load()

pca = PCA(n_components = 100)
pca.fit(train_x)

im = draw.tensor_to_image_grid(pca.components_.reshape(-1, 28, 28))
im.save("pca.png")
im.show()

output_x = pca.transform(train_x).T
output_y = one_hot(train_y, 10).T
num_dims = output_x.shape[0]
correlations = np.corrcoef(output_x, output_y)[num_dims:,:num_dims]

cor = draw.matrix_to_image(correlations)
cor.save("cor.png")
cor.show()
