#!/usr/local/bin/python

import h5py
import sys
import numpy as np
from PIL import Image

import draw

h5_file_name = sys.argv[1]
h5_grp = sys.argv[2]

image_file = None
if len(sys.argv) >= 4:
  image_file = sys.argv[3]

h5_file = h5py.File(h5_file_name, "r")
ws = np.array(h5_file[h5_grp]).transpose().reshape(-1, 28, 28)
im = draw.tensor_to_image_grid(ws)

if image_file is None:
  im.show()
else:
  im.save(image_file)
