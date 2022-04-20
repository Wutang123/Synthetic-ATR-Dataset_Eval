#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      models.py
# Create:        01/17/22
# Description:   Models used for training and testing
#---------------------------------------------------------------------

# IMPORTS:

# pytorch libraries
import torch
from torch import nn
import torch.nn.functional as F

#---------------------------------------------------------------------
# Function:    model_encoder_decoder()
# Description: Model for Encoder Decoder
#---------------------------------------------------------------------
class model_encoder_decoder(nn.Module):
    def __init__(self):
        super(model_encoder_decoder, self).__init__()

        # Encoder Model
        self.enc_conv1 = nn.Conv2d(in_channels =  3, out_channels = 32, kernel_size = (5, 5), padding = 'same')
        self.maxpool1  = nn.MaxPool2d(kernel_size = (2, 2))
        self.enc_conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), padding = 'same')
        self.maxpool2  = nn.MaxPool2d(kernel_size = (2, 2))
        self.enc_conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 'same')
        self.maxpool3  = nn.MaxPool2d(kernel_size = (2, 2))
        self.enc_conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), padding = 'same')
        self.maxpool4  = nn.MaxPool2d(kernel_size = (2, 2))

        self.flatten = nn.Flatten(start_dim = 1)

        self.linear1   = nn.Linear(in_features = 5,    out_features = 64)
        self.linear2   = nn.Linear(in_features = 64,   out_features = 64)
        self.linear3   = nn.Linear(in_features = 1088, out_features = 1024)
        self.linear4   = nn.Linear(in_features = 1024, out_features = 1024)

        self.unflatten = nn.Unflatten(dim = 1, unflattened_size=(64, 4, 4))

        # Decoder Model
        # self.dec_conv1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3))
        self.dec_conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), padding = 'same')
        self.upsample1 = nn.Upsample(scale_factor = 2)
        # self.dec_conv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3))
        self.dec_conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), padding = 'same')
        self.upsample2 = nn.Upsample(scale_factor = 2)
        # self.dec_conv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3))
        self.dec_conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3), padding = 'same')
        self.upsample3 = nn.Upsample(scale_factor = 2)
        # self.dec_conv4 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = (5, 5))
        self.dec_conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), padding = 'same')
        self.upsample4 = nn.Upsample(scale_factor = 2)
        # self.final     = nn.ConvTranspose2d(in_channels = 32, out_channels =  3, kernel_size = (3, 3))
        self.final     = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = (3, 3), padding = 'same')

    def forward(self, input, param):
        batch = input.size()[0]
        # print("input", input.size())
        # print("param", param.size(), "\n")

        # Encoding
        encoded_view = F.leaky_relu(self.enc_conv1(input))
        # print("encoded_view", encoded_view.size())
        encoded_view = self.maxpool1(encoded_view)
        # print("encoded_view", encoded_view.size())

        encoded_view = F.leaky_relu(self.enc_conv2(encoded_view))
        # print("encoded_view", encoded_view.size())
        encoded_view = self.maxpool2(encoded_view)
        # print("encoded_view", encoded_view.size())

        encoded_view = F.leaky_relu(self.enc_conv3(encoded_view))
        # print("encoded_view", encoded_view.size())
        encoded_view = self.maxpool3(encoded_view)
        # print("encoded_view", encoded_view.size())

        encoded_view = F.leaky_relu(self.enc_conv4(encoded_view))
        # print("encoded_view", encoded_view.size())
        encoded_view = self.maxpool4(encoded_view)
        # print("encoded_view", encoded_view.size(), "\n")

        # Flatten
        flatten_view = self.flatten(encoded_view)
        # print("flatten_view", flatten_view.size())

        # Linear - param
        param_view = F.leaky_relu(self.linear1(param))
        param_view = F.leaky_relu(self.linear2(param_view))
        # print(param_view.size())

        # Concatenate - param with encoder
        concatenated_view = torch.cat((flatten_view, param_view), dim = 1)
        # print(concatenated_view.size())

        # Linear
        concatenated_view = F.leaky_relu(self.linear3(concatenated_view))
        concatenated_view = F.leaky_relu(self.linear4(concatenated_view))
        # print(concatenated_view.size())

        # Unflatten
        unflatten_view = self.unflatten(concatenated_view)
        # print("unflatten_view", unflatten_view.size(), "\n")

        # Decoding
        decoded_view = F.leaky_relu(self.dec_conv1(unflatten_view))
        # print("decoded_view", decoded_view.size())
        decoded_view = self.upsample1(decoded_view)
        # print("decoded_view", decoded_view.size())

        decoded_view = F.leaky_relu(self.dec_conv2(decoded_view))
        # print("decoded_view", decoded_view.size())
        decoded_view = self.upsample2(decoded_view)
        # print("decoded_view", decoded_view.size())

        decoded_view = F.leaky_relu(self.dec_conv3(decoded_view))
        # print("decoded_view", decoded_view.size())
        decoded_view = self.upsample3(decoded_view)
        # print("decoded_view", decoded_view.size())

        decoded_view = F.leaky_relu(self.dec_conv4(decoded_view))
        # print("decoded_view", decoded_view.size())
        decoded_view = self.upsample4(decoded_view)
        # print("decoded_view", decoded_view.size())

        decoded_view = torch.tanh(self.final(decoded_view))
        # print("\ndecoded_view", decoded_view.size())

        return decoded_view