#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      helper_test_encoder_decoder.py
# Create:        03/19/22
# Description:   Evaluate Encoder Decoder Performance
#---------------------------------------------------------------------

# IMPORTS:
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
import math
import json

from Helper_Functions.models import *
from Helper_Functions.dataset import *

# pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

# sklearn libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    plot_confusion_matrix()
# Description: Plot confusion matrix
#---------------------------------------------------------------------
def plot_confusion_matrix(file, target_names, save_fig, run_path, y_label, y_pred):

    conf_mat = confusion_matrix(y_label, y_pred)
    print("Confusion Matrix: \n", conf_mat, "\n")
    file.write("Confusion Matrix: \n" + str(conf_mat) + "\n\n")

    fig = plt.figure(figsize=(10, 10))
    plt.title('Confusion matrix')
    ax = sns.heatmap(conf_mat, annot = True, cmap = plt.cm.Blues, fmt = 'g', linewidths = 1)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set(ylabel = "True Labels", xlabel = "Predicted Labels")

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if(save_fig):
        save_path = os.path.join(run_path, "Confusion_Matrix.png")
        fig.savefig(save_path, bbox_inches = 'tight')

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
    for i in range(len(target_names)):
        print(target_names[i], ": ", round(class_accuracy[i],2), "%")
        file.write(target_names[i] + ":" + str(round(class_accuracy[i],2)) + "% \n")
    print()
    file.write("\n")



#---------------------------------------------------------------------
# Function:    test_encoder_decoder()
# Description: Create syntheic images using Encoder Decoder and test using vgg16 model
#---------------------------------------------------------------------
def helper_test_encoder_decoder(args, file, run_path, number_classes, test_csv_file, vgg_model_path, ec_model_path, df_test_encoder_decoder):

    # pred_images_path = os.path.join(run_path, "pred_images")
    # os.mkdir(pred_images_path)

    batch         = args.batch
    num_worker    = args.worker
    input_size    = args.imgsz # Minimum for VGG16 is (32,32)
    num_classes   = number_classes
    save_fig      = args.save_fig
    save_data     = args.save_data

    print("Batch Size:         {}".format(batch))
    file.write("Batch Size:         {} \n".format(batch))
    print("Number of Workers:  {}".format(num_worker))
    file.write("Number of Workers:  {} \n".format(num_worker))
    print("Image Size:         {}".format(input_size))
    file.write("Image Size:         {} \n".format(input_size))

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")
    file.write("Device Used:        " + str(device) + "\n\n")

    class_id_dict = {1 : 'PICKUP',
                     2 : 'SUV'   ,
                     5 : 'BTR70' ,
                     6 : 'BRDM2' ,
                     9 : 'BMP2'  ,
                     11: 'T72'   ,
                     12: 'ZSU23' ,
                     13: '2S3'   ,
                     14: 'D20'   }
    file.write("class_id_dict: \n")
    file.write(json.dumps(class_id_dict))
    file.write("\n\n")

    # Model - VGG16
    vgg_model = models.vgg16(pretrained = False)
    num_ftrs = vgg_model.classifier[6].in_features
    vgg_model.fc = nn.Linear(num_ftrs, num_classes)
    vgg_model.load_state_dict(torch.load(vgg_model_path))
    vgg_model = vgg_model.to(device)

    # Model - Encoder-Decoder
    ec_model = model_encoder_decoder()
    ec_model = ec_model.to(device)
    ec_model.load_state_dict(torch.load(ec_model_path))
    ec_model = ec_model.to(device)

    # Dataset
    # Oraganize input encoder-decoder parameters
    image_1_parameters = df_test_encoder_decoder.drop(columns = ['image_1', 'class_id_1', 'class_type_1', 'image_2', 'class_id_2', 'class_type_2', 'time_of_day_2','range_2', 'orientation_2'])
    image_1_parameters['time_of_day_1'].replace(to_replace = 'Day', value = 1, inplace = True)
    image_1_parameters['time_of_day_1'].replace(to_replace = 'Night', value = 0, inplace = True)
    image_1_parameters = image_1_parameters.to_numpy()
    # print(image_1_parameters)

    image_2_parameters = df_test_encoder_decoder.drop(columns = ['image_1', 'class_id_1', 'class_type_1', 'time_of_day_1','range_1', 'orientation_1', 'image_2', 'class_id_2', 'class_type_2'])
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

    # print(df_test_encoder_decoder.info())
    channel = list(input_size)
    channel.insert(0, 3)
    channel = tuple(channel)
    flip_channel = list(input_size)
    flip_channel.insert(2, 3)
    flip_channel = tuple(flip_channel)

    testing_set            = Encoder_Decoder_Dataset(df_test_encoder_decoder, input_size, channel)
    test_loader            = DataLoader(testing_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    param_test_loader      = DataLoader(param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    zero_param_test_loader = DataLoader(zero_param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)

    # Test Model
    vgg_model.eval()
    ec_model.eval()

    y_label    = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred     = torch.zeros(0, dtype = torch.long, device = 'cpu')
    # pred_images_list = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        params_iter      = iter(param_test_loader)
        zero_params_iter = iter(zero_param_test_loader)

        for batch, (images, labels) in enumerate(test_loader):
            if (batch == 0) or (batch % 100 == 0):
                print("*" * 10)
                file.write("*" * 10 +"\n")
                print("BATCH {}:".format(batch))

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

            # Create synthetic images using encoder decoder
            ec_outputs = ec_model(images_1, params, zero_params)
            # print(pred.size())
            # pred_images_list = torch.cat([pred_images_list, pred.cpu()])
            # print(pred_images_list.size())

            vgg_outputs  = vgg_model(ec_outputs)
            _, vgg_preds = torch.max(vgg_outputs, 1)

            # Append batch prediction results
            y_label    = torch.cat([y_label, labels_2.view(-1).cpu()])
            y_pred     = torch.cat([y_pred, vgg_preds.view(-1).cpu()])

    print("*" * 10 + "\n")
    file.write("*" * 10 +"\n\n")

    y_label = y_label.numpy()
    y_pred  = y_pred.numpy()

    # print(pred_images_list)
    # print(pred_images_list.size())
    # file.write(str(pred_images_list.size()) + "\n\n")

    if(save_data):
        df_test_csv = pd.read_csv(test_csv_file)
        test_image = df_test_csv['image_2']

        # Open csv file in 'w' mode
        with open(os.path.join(run_path, "test_encoder_decoder_data.csv"), 'w', newline ='') as csv_file:
            length = len(y_label)

            write = csv.writer(csv_file)
            write.writerow(["image", "ground_truth", "predict"])
            for i in range(length):
                write.writerow([test_image[i], class_id_dict[y_label[i]], class_id_dict[y_pred[i]]])

    target_names = ['PICKUP','SUV','BTR70','BRDM2','BMP2','T72','ZSU23', '2S3', 'D20']

    # Accuracy Score
    acc_score = accuracy_score(y_label, y_pred)
    print("Model Accuracy: ", acc_score, "\n")
    file.write("Model Accuracy: " + str(acc_score) + "\n\n")

    plot_confusion_matrix(file, target_names, save_fig, run_path, y_label, y_pred)


#=====================================================================