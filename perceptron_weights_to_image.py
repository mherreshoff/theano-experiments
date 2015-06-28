#!/usr/local/bin/python

import h5py
import sys
import numpy as np
from PIL import Image

prog_name, h5_file_name, img_file_name = sys.argv

h5_file = h5py.File(h5_file_name, "r")

dset = h5_file["W1"]
print dset[1:].shape
ws = np.array(dset)[1:].transpose().reshape(-1, 28, 28)
ws_norm = ws / np.abs(ws).max()

# Note: assumes 100 hidden units for now.
image = ws_norm.reshape(10, 10, 28, 28).swapaxes(1, 2).reshape(-1, 10*28)

red = np.maximum(image, 0)*255.0
green = np.zeros_like(image)
blue = -np.minimum(image, 0)*255.0
print red.shape
print green.shape
print blue.shape

im = Image.fromarray(np.uint8(np.dstack((red,green,blue))))

im.save(img_file_name)
