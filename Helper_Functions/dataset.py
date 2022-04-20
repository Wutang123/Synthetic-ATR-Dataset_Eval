#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      dataset.py
# Create:        03/19/22
# Description:   Create dataset
#---------------------------------------------------------------------

# IMPORTS:
from numpy import save
import scipy.io # Open .mat files
import cv2
import numpy as np

# pytorch libraries
import torch
from torch.utils.data import Dataset

# Class:
#---------------------------------------------------------------------
# Function:    Encoder_Decoder_Dataset()
# Description: Characterizes ATR Dataset for PyTorch - Encoder & Decoder
#---------------------------------------------------------------------
class Encoder_Decoder_Dataset(Dataset):
    def __init__(self, df, resize, channel):
        self.images_1 = df['image_1']
        self.ids_1    = df['class_id_1']
        self.images_2 = df['image_2']
        self.ids_2    = df['class_id_2']
        self.resize = resize
        self.new_channel = channel

        # print(self.images_1)
        # print(self.ids_1)
        # print(self.images_2)
        # print(self.ids_2)
        # print(self.resize)
        # print(self.new_channel)

    def __getitem__(self, index):
        # keys = x.keys()
        # print(keys)
        # dict_keys(['__header__', '__version__', '__globals__', 'target_chip'])
        image_1 = scipy.io.loadmat(self.images_1[index], squeeze_me = True)
        image_2 = scipy.io.loadmat(self.images_2[index], squeeze_me = True)

        image_1 = image_1['target_chip']/65535
        image_2 = image_2['target_chip']/65535

        image_1 = np.float32(image_1)
        image_2 = np.float32(image_2)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)

        label_1 = torch.tensor(int(self.ids_1[index]))
        label_2 = torch.tensor(int(self.ids_2[index]))

        if self.resize:
            image_1 = cv2.resize(image_1, self.resize)
            image_1 = torch.from_numpy(image_1).float()
            image_1 = image_1.reshape((self.new_channel))

            image_2 = cv2.resize(image_2, self.resize)
            image_2 = torch.from_numpy(image_2).float()
            image_2 = image_2.reshape((self.new_channel))

        return [image_1, image_2], [label_1, label_2]

    def __len__(self):
        len_image_1 = len(self.images_1)
        len_image_2 = len(self.images_2)

        if(len_image_1 == len_image_2):
            return len_image_1
        else:
            return min(len_image_1, len_image_2)



#---------------------------------------------------------------------
# Function:    Classifier_Dataset()
# Description: Characterizes ATR Dataset for PyTorch - Classifer
#---------------------------------------------------------------------
class Classifier_Dataset(Dataset):
    def __init__(self, df, resize, channel):
        self.images      = df['image']
        self.ids         = df['class_id']
        self.resize      = resize
        self.new_channel = channel

        # print(self.images)
        # print(self.ids)
        # print(self.resize)
        # print(self.new_channel)

    def __getitem__(self, index):
        image = scipy.io.loadmat(self.images[index], squeeze_me = True)
        # keys = x.keys()
        # print(keys)
        # dict_keys(['__header__', '__version__', '__globals__', 'target_chip'])
        image = image['target_chip']/65535
        label = torch.tensor(int(self.ids[index]))

        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.resize:
            image = cv2.resize(image, self.resize)
            image = torch.from_numpy(image).float()
            image = image.reshape((self.new_channel))

        return image, label

    def __len__(self):
        return len(self.images)