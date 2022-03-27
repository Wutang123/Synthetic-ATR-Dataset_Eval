#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_encoder_decoder.py
# Create:        01/17/22
# Description:   Train Encoder Decoder
#---------------------------------------------------------------------

# IMPORTS:
import os
import sys
import time
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

from Helper_Functions.models import *
from Helper_Functions.dataset import *

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train()
# Description: Train model
#---------------------------------------------------------------------
def train(epoch, epoch_num, flip_channel, run_path, file, train_loader, param_train_loader, zero_param_train_loader, model, loss_function, optimizer, device):
    model.train()

    size        = len(train_loader.dataset)
    num_batches = len(train_loader)

    params_iter      = iter(param_train_loader)
    zero_params_iter = iter(zero_param_train_loader)

    train_loss = 0

    for batch, (images, labels) in enumerate(train_loader):
        images_1 = images[0]
        images_2 = images[1]
        labels_1 = labels[0]
        labels_2 = labels[1]

        params_next       = next(params_iter)
        zero_params_next  = next(zero_params_iter)

        images_1    = images_1.to(device)
        images_2    = images_2.to(device)
        labels_1    = labels_1.to(device)
        labels_2    = labels_2.to(device)
        params      = params_next.to(device)
        zero_params = zero_params_next.to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        pred = model(images_1, params, zero_params)

        # Sample image
        if batch == 0 and epoch == (epoch_num - 1):

            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(1, 3, 1)
            plt.title("images_1[0]")
            # print(images_1[0], type(images_1[0]), images_1[0].size())
            # a = images_1[0].squeeze()
            # print(a.size())
            # a = a.detach().to("cpu").numpy()
            a = images_1[0]
            a = a.reshape((flip_channel))
            a = a.detach().to("cpu").numpy()
            # print(type(a))
            # print(a.shape)
            plt.imshow(a, cmap = 'gray')
            # print()

            fig.add_subplot(1, 3, 2)
            plt.title("images_2[0]")
            # print(images_2[0], type(images_2[0]), images_2[0].size())
            # b = images_2[0].squeeze()
            # print(b.size())
            # b = b.detach().to("cpu").numpy()
            b = images_2[0]
            b = b.reshape((flip_channel))
            b = b.detach().to("cpu").numpy()
            # print(type(b))
            # print(b.shape)
            plt.imshow(b, cmap = 'gray')
            # print()

            fig.add_subplot(1, 3, 3)
            plt.title("pred[0]")
            # print(pred[0], type(pred[0]), pred[0].size())
            # c = pred[0].squeeze()
            # print(c.size())
            # c = c.detach().to("cpu").numpy()
            c = pred[0]
            c = c.reshape((flip_channel))
            c = c.detach().to("cpu").numpy()
            # print(type(c))
            # print(c.shape)
            plt.imshow(c, cmap = 'gray')
            # print()

            save_path = os.path.join(run_path, (str(epoch) + "_images_1[0]_images_2[0]_pred[0].png"))
            plt.savefig(save_path, bbox_inches = 'tight')

        # Calculate the loss
        loss = loss_function(pred, images_2)

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        train_loss += loss.item() * images_1.size(0)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # break

    train_loss /= num_batches
    print(f"Avg loss: {train_loss:>8f}")
    file.write("Training Error: [Avg Loss: {:>8f}]\n".format(train_loss))

    return train_loss



#---------------------------------------------------------------------
# Function:    calcResults()
# Description: Print and write all results
#---------------------------------------------------------------------
def calcResults(file, save_data, run_path, total_train_loss, total_time_list):

    # Print Data
    print("Training Loss:       ", total_train_loss)
    file.write("Training Loss:       " + str(total_train_loss) + "\n")
    print("Total Time (sec):    ", total_time_list)
    file.write("Total Time (sec):    " + str(total_time_list) + "\n\n")
    print()
    print("Average Training Loss:       {}".format(np.mean(total_train_loss)))
    file.write("Average Training Loss:       {} \n".format(np.mean(total_train_loss)))
    print("Average Total Time (sec):    {}".format(np.mean(total_time_list)))
    file.write("Average Total Time (sec):    {} \n\n".format(np.mean(total_time_list)))
    print()

    if(save_data):
        # Open csv file in 'w' mode
        with open(os.path.join(run_path, "train_classifier_data.csv"), 'w', newline ='') as csv_file:
            length = len(total_train_loss)

            write = csv.writer(csv_file)
            write.writerow(["total_train_loss", "total_time_list"])
            for i in range(length):
                write.writerow([total_train_loss[i], total_time_list[i]])



#---------------------------------------------------------------------
# Function:    plotFigures()
# Description: Plot comparsion plots
#---------------------------------------------------------------------
def plotFigures(save_fig, run_path, total_train_loss):

    fig = plt.figure()
    plt.plot(total_train_loss, label = 'Training Loss')
    plt.legend()
    if (save_fig):
        save_path = os.path.join(run_path, "Training_Loss.png")
        fig.savefig(save_path, bbox_inches = 'tight')

#---------------------------------------------------------------------
# Function:    train_encoder_decoder()
# Description: Train Encoder Decoder
#---------------------------------------------------------------------
def helper_train_encoder_decoder(args, file, run_path, number_classes, df_train_encoder_decoder):

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

    # Model
    model = model_encoder_decoder()
    model = model.to(device) # Add device to model

    # Dataset
    # Oraganize input encoder-decoder parameters
    image_1_parameters = df_train_encoder_decoder.drop(columns = ['image_1', 'class_id_1', 'class_type_1', 'image_2', 'class_id_2', 'class_type_2', 'time_of_day_2','range_2', 'orientation_2'])
    image_1_parameters['time_of_day_1'].replace(to_replace = 'Day', value = 1, inplace = True)
    image_1_parameters['time_of_day_1'].replace(to_replace = 'Night', value = 0, inplace = True)
    image_1_parameters = image_1_parameters.to_numpy()
    # print(image_1_parameters)

    image_2_parameters = df_train_encoder_decoder.drop(columns = ['image_1', 'class_id_1', 'class_type_1', 'time_of_day_1','range_1', 'orientation_1', 'image_2', 'class_id_2', 'class_type_2'])
    image_2_parameters['time_of_day_2'].replace(to_replace = 'Day', value = 1, inplace = True)
    image_2_parameters['time_of_day_2'].replace(to_replace = 'Night', value = 0, inplace = True)
    image_2_parameters = image_2_parameters.to_numpy()
    # print(image_2_parameters)

    # Open Az_dict
    az_dict_file = open("Helper_Functions/az_dict.txt", 'r')
    az_dict = {}
    for line in az_dict_file:
        key, value = line.strip().split(' ')
        az_dict[int(key.strip())] = int(value.strip())
    az_dict_file.close()

    rad_factor = 100.0
    param      = []
    zero_param = []
    for i in range(len(image_2_parameters)):
        param_range   = float(image_2_parameters[i][1])
        param_az      = float(az_dict.get(image_2_parameters[i][2]))
        param_daytime = float(image_2_parameters[i][0])

        param.append([param_range / rad_factor,
                      math.sin(math.radians(param_daytime)),
                      math.cos(math.radians(param_daytime)),
                      math.sin(math.radians(param_az)),
                      math.cos(math.radians(param_az))])

        zero_param.append([0,
                           math.sin(math.radians(0)),
                           math.cos(math.radians(0)),
                           math.sin(math.radians(0)),
                           math.cos(math.radians(0))])

    param      = torch.FloatTensor(param)
    zero_param = torch.FloatTensor(zero_param)
    # print(param)
    # print(zero_param)

    # print(df_train_encoder_decoder.info())
    channel = list(input_size)
    channel.insert(0, 3)
    channel = tuple(channel)
    flip_channel = list(input_size)
    flip_channel.insert(2, 3)
    flip_channel = tuple(flip_channel)

    training_set            = Encoder_Decoder_Dataset(df_train_encoder_decoder, input_size, channel)
    train_loader            = DataLoader(training_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    param_train_loader      = DataLoader(param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    zero_param_train_loader = DataLoader(zero_param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)

    # Loss and Optimizer Function
    loss_function = nn.MSELoss().to(device)
    optimizer     = optim.Adam(model.parameters(), lr = learning_rate)

    # Verify Tensor Size, should be [batch_size, channels, image_height, image_width] (e.g [64, 3, 64, 64])
    first_train = 1
    train_size_1 = None
    train_size_2 = None
    for i, (images, labels) in enumerate(train_loader):
        images_1 = images[0]
        images_2 = images[1]
        labels_1 = labels[0]
        labels_2 = labels[1]

        if(first_train):
            train_size_1 = images_1.shape
            train_size_2 = images_2.shape
            first_train = 0

            # print(i, "images_1.shape, images_1.dtype, images_1.device: ", images_1.shape, images_1.dtype, images_1.device)
            # print(i, "images_2.shape, images_2.dtype, images_2.device: ", images_2.shape, images_2.dtype, images_2.device)
            # print(i, "labels_1.shape, labels_1.dtype, labels_1.device: ", labels_1.shape, labels_1.dtype, labels_1.device)
            # print(i, "labels_2.shape, labels_2.dtype, labels_2.device: ", labels_2.shape, labels_2.dtype, labels_2.device)
        else:
            if(images_1.shape != train_size_1):
                print("ERROR: Mismatch train_loader(train_size_1) Size!")
                file.write("ERROR: Mismatch train_loader(train_size_1) Size!\n")
                sys.exit()

            if(images_2.shape != train_size_2):
                print("ERROR: Mismatch train_loader(train_size_2) Size!")
                file.write("ERROR: Mismatch train_loader(train_size_2) Size!\n")
                sys.exit()

    # Summary of Model
    print("ENCODER: Tensor Image Size [batch_size, channels, image_height, image_width]: ", train_size_1)
    file.write("\nENCODER: Tensor Image Size [batch_size, channels, image_height, image_width]:: " + str(train_size_1) + "\n")
    print("DECODER: Tensor Image Size [batch_size, channels, image_height, image_width]: ", train_size_2, "\n")
    file.write("\nDECODER: Tensor Image Size [batch_size, channels, image_height, image_width]: " + str(train_size_2) + "\n\n")
    with redirect_stdout(file):
        print(model)
    file.write("\n")

    # Main Training Function
    total_train_loss     = []
    total_time_list      = []

    for epoch in range(epoch_num):
        start_time = time.time()

        print("*" * 10)
        file.write("*" * 10 +"\n")
        print("EPOCH {}:".format(epoch))
        file.write("EPOCH: " + str(epoch) + "\n")

        train_loss = train(epoch, epoch_num, flip_channel, run_path, file, train_loader, param_train_loader, zero_param_train_loader, model, loss_function, optimizer, device)
        total_train_loss.append(train_loss) # Average Training Loss

        end_time = time.time()
        total_time_difference = end_time - start_time
        total_time_list.append(total_time_difference)

    print("*" * 10 + "\n")
    file.write("*" * 10 +"\n\n")

    # Save Model
    if(save_model):
        save_path = os.path.join(run_path, "encoder_decoder.pth")
        torch.save(model.state_dict(), save_path)

     # Organize Results
    calcResults(file, save_data, run_path, total_train_loss, total_time_list)
    plotFigures(save_fig, run_path, total_train_loss)

#=====================================================================