#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_classifier.py
# Create:        01/17/22
# Description:   Train Classifier
#---------------------------------------------------------------------

# IMPORTS:
import sys
from torchsummary import summary

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train_classifier()
# Description: Train Classifier
#---------------------------------------------------------------------
def train_classifier(df_train, df_test):

    # TODO: Update Later
    learning_rate = 0.0001
    batch =  64
    num_worker = 4
    epoch_num = 3
    use_pretrained = True
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    num_classes = 9

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")

    # TODO: Add loading model feature and select different models option
    # VGG16
    model = models.vgg16(pretrained = use_pretrained)
    num_ftrs = model.classifier[6].in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device) # Add device to model

    # Dataset
    training_set = Dataset(df_train)
    train_loader = DataLoader(training_set, batch_size = batch, shuffle = True, num_workers = num_worker, drop_last = True)
    test_set    = Dataset(df_test)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = True)

    # Loss and Optimizer Function
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer    = optim.Adam(model.parameters(), lr = learning_rate)

    # Verify Tensor Size, should be [batch_size, channel_size, image_height, image_width] (e.g [32, 3, 225, 225])
    first_train = 1
    train_size = None
    for i, (images, labels) in enumerate(train_loader):
        if(first_train):
            train_size = images.shape
            first_train = 0
        else:
            if(images.shape != train_size):
                print("ERROR: Mismatch train_loader Size!")
                sys.exit()

    first_test = 1
    test_size = None
    for i, (images, labels) in enumerate(test_loader):
        if(first_test):
            test_size = images.shape
            first_test = 0
        else:
            if(images.shape != test_size):
                print("ERROR: Mismatch test_loader Size!")
                sys.exit()

    # Summary of Model
    print("Tensor Image Size [batch_size, channel_size, image_height, image_width]: ", train_size)
    summary(model, input_size = (train_size[1], train_size[2], train_size[3]))
#=====================================================================