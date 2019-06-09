from __future__ import absolute_import
import sys
sys.path.append('./')

import time
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
from unittest import TestCase

from tps_spatial_transformer import TPSSpatialTransformer


class TestTPSSpatialTransformer(TestCase):

  control_points_per_side = 6

  def test_forward(self):

    # specify the input control points with the shape [3, 6, 2]
    test_input_ctrl_pts = np.array([
      [
        [-0.1, 0.4], [0.5, 0.1], [0.9, 0.4],
        [-0.1, 0.9], [0.5, 0.6], [0.9, 0.9]
      ],
      [
        [0.1, 0.1], [0.5, 0.4], [0.9, 0.1],
        [0.1, 0.6], [0.5, 0.9], [0.9, 0.6]
      ],
      [
        [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
        [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
      ]
    ], dtype=np.float32)
    source_control_points = torch.from_numpy(test_input_ctrl_pts)

    # preprocess the input image to match the pytorch's format.
    test_im = Image.open('./test_image.jpg').resize((128, 128))
    test_image_array = np.array(test_im).astype('float32')
    test_image_array = np.array([test_image_array, test_image_array, test_image_array])
    test_images = torch.from_numpy(test_image_array)
    test_images = test_images.permute((0,3,1,2))

    target_height, target_width = 32, 100
    margin = 0.05

    print('initialize module')
    beg_time = time.time()
    tps = TPSSpatialTransformer(
      output_image_size=(target_height, target_width),
      num_control_points=6,
      margins=(margin, margin))
    past_time = time.time() - beg_time
    print('initialization takes %.02fs' % past_time)

    target_control_points = tps.target_control_points
    target_control_points_array = target_control_points.numpy()
    target_image, source_coordinate = tps(test_images, source_control_points)
    source_coordinate_array = source_coordinate.numpy()
    grid = source_coordinate.view(-1, target_height, target_width, 2)
    target_image = target_image.permute((0,2,3,1)).numpy()

    if True:
      plt.figure()
      plt.subplot(3,4,1)
      plt.scatter(test_input_ctrl_pts[0,:,0], test_input_ctrl_pts[0,:,1])
      plt.subplot(3,4,2)
      plt.scatter(target_control_points_array[:,0], target_control_points_array[:,1])
      plt.subplot(3,4,3)
      plt.scatter(source_coordinate_array[0,:,0], source_coordinate_array[0,:,1], marker='+')
      plt.subplot(3,4,4)
      plt.imshow(target_image[0].astype(np.uint8))

      plt.subplot(3,4,5)
      plt.scatter(test_input_ctrl_pts[1,:,0], test_input_ctrl_pts[1,:,1])
      plt.subplot(3,4,6)
      plt.scatter(target_control_points_array[:,0], target_control_points_array[:,1])
      plt.subplot(3,4,7)
      plt.scatter(source_coordinate_array[1,:,0], source_coordinate_array[1,:,1], marker='+')
      plt.subplot(3,4,8)
      plt.imshow(target_image[1].astype(np.uint8))

      plt.subplot(3,4,9)
      plt.scatter(test_input_ctrl_pts[2,:,0], test_input_ctrl_pts[2,:,1])
      plt.subplot(3,4,10)
      plt.scatter(target_control_points_array[:,0], target_control_points_array[:,1])
      plt.subplot(3,4,11)
      plt.scatter(source_coordinate_array[2,:,0], source_coordinate_array[2,:,1], marker='+')
      plt.subplot(3,4,12)
      plt.imshow(target_image[2].astype(np.uint8))

      plt.show()
      plt.savefig('plot.png')

if __name__ == '__main__':
  test_tps = TestTPSSpatialTransformer()
  test_tps.test_forward()
