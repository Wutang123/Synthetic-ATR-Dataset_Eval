#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      sample_images.py
# Create:        02/27/22
# Description:   Save a subset of images
#---------------------------------------------------------------------

# IMPORTS:
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
import numpy as np
import torch

def main():
    class_dict = {
                    0 : 'PICKUP',
                    1 : 'SUV'   ,
                    2 : 'BTR70' ,
                    3 : 'BRDM2' ,
                    4 : 'BMP2'  ,
                    5 : 'T72'   ,
                    6 : 'ZSU23' ,
                    7 : '2S3'   ,
                    8 : 'D20'   }

    input_images = ['INPUT\Dataset\cegr1923\cegr01923_0001_1_PICKUP.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0002_1_SUV.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0005_1_BTR70.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0006_1_BRDM2.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0009_1_BMP2.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0011_1_T72.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0012_1_ZSU23.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0013_1_2S3.mat',
                    'INPUT\Dataset\cegr1923\cegr01923_0014_1_D20.mat']


    image = scipy.io.loadmat(input_images[0], squeeze_me = True)
    image = image['target_chip']/65535
    print(image)
    print(image.shape)
    print()

    image = np.float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(image)
    print(image.shape)
    print()

    image = cv2.resize(image, (64, 64))
    image = torch.from_numpy(image).float()
    print(image)
    print(image.shape)
    print()

    image = image.reshape((3, 64, 64))
    print(image)
    print(image.shape)
    print()


    # for i in range(len(input_images)):
    #     mat = scipy.io.loadmat(input_images[i], squeeze_me = True)
    #     mat = mat['target_chip']

    #     fig = plt.figure(figsize=(10, 7))
    #     fig.add_subplot(2, 2, 1)
    #     plt.title("originial")
    #     plt.imshow(mat, cmap = 'gray')
    #     print(mat)
    #     print(mat.shape)
    #     print()

    #     fig.add_subplot(2, 2, 2)
    #     plt.title("normalize")
    #     mat = mat/65535
    #     plt.imshow(mat, cmap = 'gray')
    #     print(mat)
    #     print(mat.shape)
    #     print()

    #     fig.add_subplot(2, 2, 3)
    #     plt.title("3 dim - 2")
    #     mat_2 = np.float32(mat)
    #     mat_2 = cv2.cvtColor(mat_2, cv2.COLOR_GRAY2RGB)
    #     plt.imshow(mat_2, cmap = 'gray')
    #     print(mat_2)
    #     print(mat_2.shape)
    #     print()

    #     fig.add_subplot(2, 2, 4)
    #     plt.title("3 dim - 2")
    #     mat_3 = mat_2.reshape(3, 20, 40)
    #     plt.imshow(mat_3, cmap = 'gray')
    #     print(mat_3)
    #     print(mat_3.shape)
    #     print()

    #     plt.show()


    # for i in range(len(input_images)):
    #     mat = scipy.io.loadmat(input_images[i], squeeze_me=True)
    #     mat = mat['target_chip']

    #     img = plt.imshow(mat, cmap = 'gray')
    #     plt.title(str(class_dict[i]) + " - 20 by 40")
    #     save_path = os.path.join("OUTPUT", "sample_images", "sample_image_" + str(class_dict[i]) + ".png")
    #     plt.savefig(save_path, bbox_inches = 'tight')

    #     mat_resize = cv2.resize(mat, (80, 40))
    #     plt.title(str(class_dict[i]) + " - 40 by 80")
    #     img_resize = plt.imshow(mat_resize, cmap = 'gray')
    #     save_path = os.path.join("OUTPUT", "sample_images", "sample_image_resized_" + str(class_dict[i]) + ".png")
    #     plt.savefig(save_path, bbox_inches = 'tight')

# MODULES:
if __name__ == "__main__":
    main()