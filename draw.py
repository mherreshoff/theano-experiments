import numpy as np
from PIL import Image

def matrix_to_image(m):
  m_norm = m / np.abs(m).max()
  red = np.maximum(m_norm, 0)*255.0
  green = np.zeros_like(m_norm)
  blue = -np.minimum(m_norm, 0)*255.0
  return Image.fromarray(np.uint8(np.dstack((red, green, blue))))

def image_grid(pics, cols=10, margin=2, bg_color="white"):
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

def tensor_to_image_grid(np3d):
  images = [matrix_to_image(m) for m in np3d]
  return image_grid(images)
