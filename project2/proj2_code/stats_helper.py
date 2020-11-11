import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from PIL import ImageOps
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  main = os.listdir(dir_name)
  scaler = StandardScaler()

  for j in range(len(main)):
      directory = dir_name + main[j]
      #print(directory)
      sub = os.listdir(directory)
      for i in range(len(sub)):
          img_loc = directory + '/' + sub[i] + '/*'
          for f in glob.iglob(img_loc):
              grayscale_img = np.asarray(ImageOps.grayscale(Image.open(f)))
              scaled_img = grayscale_img / 255
              flattened_img = scaled_img.flatten()
              final_img = np.reshape(flattened_img, (-1, 1))
              scaled_data = scaler.partial_fit(np.array(final_img))

  mean = scaled_data.mean_
  std = scaled_data.scale_
  
  ############################################################################
  # raise NotImplementedError('compute_mean_and_std not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
