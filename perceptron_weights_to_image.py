#!/usr/local/bin/python

import h5py
import sys
import numpy as np
from PIL import Image

def weights_to_pic(w):
  w_norm = w / np.abs(w).max()
  red = np.maximum(w_norm, 0)*255.0
  green = np.zeros_like(w_norm)
  blue = -np.minimum(w_norm, 0)*255.0
  return Image.fromarray(np.uint8(np.dstack((red, green, blue))))

def make_grid(pics, cols=10, margin=2, bg_color="white"):
  """aranges pics in a grid with white margins.
  Assumes all pics are the same size."""
  xsize, ysize = pics[0].size
  rows = (len(pics) + cols - 1) / cols
  im = Image.new("RGB", ((2*margin+xsize)*cols, (2*margin+ysize)*rows), bg_color)
  for i, pic in enumerate(pics):
    col = i % cols
    row = i / cols
    im.paste(pic, ((2*margin+xsize)*col+margin, (2*margin+ysize)*row+margin))
  return im

h5_file_name = sys.argv[1]
h5_grp = sys.argv[2]

image_file = None
if len(sys.argv) >= 4:
  image_file = sys.argv[3]

h5_file = h5py.File(h5_file_name, "r")
ws = np.array(h5_file[h5_grp]).transpose().reshape(-1, 28, 28)
im = make_grid([weights_to_pic(w) for w in ws])

if image_file is None:
  im.show()
else:
  im.save(image_file)
