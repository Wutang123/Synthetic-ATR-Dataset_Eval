#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      helper_train_classifier.py
# Create:        01/17/22
# Description:   Train Classifier
#---------------------------------------------------------------------

# IMPORTS:
import os
import sys
import time
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import csv

from Helper_Functions.dataset import *

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train()
# Description: Train model
#---------------------------------------------------------------------
def train(file, train_loader, model, loss_function, optimizer, device):
    model.train()

    size        = len(train_loader.dataset)
    num_batches = len(train_loader)

    train_loss     = 0
    train_accuracy = 0

    for batch, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Compute prediction
        pred = model(images)
        # print(pred)
        # print(pred.size())
        # print(labels)
        # print(labels.size())

        # Calculate the loss
        loss = loss_function(pred, labels)

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # break

    train_loss /= num_batches
    train_accuracy /= size
    print(f"Training Error: [Accuracy: {(100 * train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}]")
    file.write("Training Error: [Accuracy: {:>0.5f}%, Avg Loss: {:>8f}]\n".format((100 * train_accuracy), train_loss))

    return train_loss, train_accuracy



#---------------------------------------------------------------------
# Function:    calcResults()
# Description: Print and write all results
#---------------------------------------------------------------------
def calcResults(file, save_data, run_path, total_train_loss, total_train_accuracy, total_time_list):

    # Print Data
    print("Training Loss:       ", total_train_loss)
    file.write("Training Loss:       " + str(total_train_loss) + "\n")
    print("Training Accuracy:   ", total_train_accuracy)
    file.write("Training Accuracy:   " + str(total_train_accuracy) + "\n")
    print("Total Time (sec):    ", total_time_list)
    file.write("Total Time (sec):    " + str(total_time_list) + "\n\n")
    print()

    print("Best Training Accuracy: ", max(total_train_accuracy), " at Epoch: ", total_train_accuracy.index(max(total_train_accuracy)))
    file.write("Best Training Accuracy: " + str(max(total_train_accuracy)) + " at Epoch: " + str(total_train_accuracy.index(max(total_train_accuracy))) + "\n\n")
    print()

    print("Average Training Loss:       {}".format(np.mean(total_train_loss)))
    file.write("Average Training Loss:       {} \n".format(np.mean(total_train_loss)))
    print("Average Training Accuracy:   {}".format(np.mean(total_train_accuracy)))
    file.write("Average Training Accuracy:   {} \n".format(np.mean(total_train_accuracy)))
    print("Average Total Time (sec):    {}".format(np.mean(total_time_list)))
    file.write("Average Total Time (sec):    {} \n\n".format(np.mean(total_time_list)))
    print()

    if(save_data):
        # Open csv file in 'w' mode
        with open(os.path.join(run_path, "train_classifier_data.csv"), 'w', newline ='') as csv_file:
            length = len(total_train_loss)

            write = csv.writer(csv_file)
            write.writerow(["total_train_loss", "total_train_accuracy", "total_time_list"])
            for i in range(length):
                write.writerow([total_train_loss[i], total_train_accuracy[i], total_time_list[i]])



#---------------------------------------------------------------------
# Function:    plotFigures()
# Description: Plot comparsion plots
#---------------------------------------------------------------------
def plotFigures(save_fig, run_path, total_train_loss, total_train_accuracy):

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(run_path, "Training_Loss_vs_Training_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(run_path, "Training_Loss.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    fig = plt.figure()
    plt.plot(total_train_accuracy, label = 'Training Accuracy')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(run_path, "Training_Accuracy.png")
        fig.savefig(save_path, bbox_inches = 'tight')

#---------------------------------------------------------------------
# Function:    helper_train_classifier()
# Description: Train Classifier
#---------------------------------------------------------------------
def helper_train_classifier(args, file, run_path, number_classes, df_train_classifier):

    learning_rate = args.lr
    batch         = args.batch
    num_worker    = args.worker
    epoch_num     = args.epoch
    input_size    = args.imgsz # Minimum for VGG16 is (32,32)
    num_classes   = number_classes
    save_model    = args.save_model
    save_fig      = args.save_fig
    save_data     = args.save_data

    print("Learning Rate:      {}".format(learning_rate))
    file.write("Learing Rate:       {} \n".format(learning_rate))
    print("Batch Size:         {}".format(batch))
    file.write("Batch Size:         {} \n".format(batch))
    print("Number of Workers:  {}".format(num_worker))
    file.write("Number of Workers:  {} \n".format(num_worker))
    print("Epoch Number:       {}".format(epoch_num))
    file.write("Epoch Number:       {} \n".format(epoch_num))
    print("Image Size:         {}".format(input_size))
    file.write("Image Size:         {} \n".format(input_size))

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")
    file.write("Device Used:        " + str(device) + "\n")

    # Model - VGG16
    model = models.vgg16(pretrained = False)

    # Update number of output classes from 1000 to 9
    num_ftrs = model.classifier[6].in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # model.classifier[6] = nn.Linear(num_ftrs, num_classes + 6) # Classes: 1, 2, 5, 6, 9, 11, 12, 13, 14 [Total: 15]
    model = model.to(device) # Add device to model

    # Dataset
    # print(df_train_classifier.info())
    channel = list(input_size)
    channel.insert(0, 3)
    channel = tuple(channel)

    training_set = Classifier_Dataset(df_train_classifier, input_size, channel)
    train_loader = DataLoader(training_set, batch_size = batch, shuffle = True, num_workers = num_worker, drop_last = True)

    # Loss and Optimizer Function
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer     = optim.Adam(model.parameters(), lr = learning_rate)

    # Verify Tensor Size, should be [batch_size, channels, image_height, image_width] (e.g [64, 3, 64, 64])
    first_train = 1
    train_size = None
    for i, (images, labels) in enumerate(train_loader):
        if(first_train):
            train_size = images.shape
            first_train = 0
            # print(images.shape, images.dtype, images.device)
            # print(labels.shape, print(labels.dtype, print(labels.device)
        else:
            if(images.shape != train_size):
                print("ERROR: Mismatch train_loader Size!")
                file.write("ERROR: Mismatch train_loader Size!\n")
                sys.exit()

    # Summary of Model
    print("Tensor Image Size [batch_size, channels, image_height, image_width]: ", train_size, "\n")
    file.write("\nTensor Image Size [batch_size, channel_size, image_height, image_width]: " + str(train_size) + "\n\n")
    summary(model, input_size = (train_size[1], train_size[2], train_size[3]))
    print()
    with redirect_stdout(file):
        summary(model, input_size = (train_size[1], train_size[2], train_size[3]))
    file.write("\n")

    # Main Training Function
    total_train_loss     = []
    total_train_accuracy = []
    total_time_list      = []

    for epoch in range(epoch_num):
        start_time = time.time()

        print("*" * 10)
        file.write("*" * 10 +"\n")
        print("EPOCH {}:".format(epoch))
        file.write("EPOCH: " + str(epoch) + "\n")

        train_loss, train_accuracy = train(file, train_loader, model, loss_function, optimizer, device)
        total_train_loss.append(train_loss)         # Average Training Loss
        total_train_accuracy.append(train_accuracy) # Average Training Accuracy

        end_time = time.time()
        total_time_difference = end_time - start_time
        total_time_list.append(total_time_difference)

    print("*" * 10 + "\n")
    file.write("*" * 10 +"\n\n")

    # Save Model
    if(save_model):
        save_path = os.path.join(run_path, "classifier.pth")
        torch.save(model.state_dict(), save_path)

    # Organize Results
    calcResults(file, save_data, run_path, total_train_loss, total_train_accuracy, total_time_list)
    plotFigures(save_fig, run_path, total_train_loss, total_train_accuracy)

#=====================================================================