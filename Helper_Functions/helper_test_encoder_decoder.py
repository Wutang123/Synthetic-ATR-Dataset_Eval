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
import sys

from Helper_Functions.models import *
from Helper_Functions.dataset import *

# pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

# sklearn libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    plot_confusion_matrix()
# Description: Plot confusion matrix
#---------------------------------------------------------------------
def plot_confusion_matrix(file, target_names, save_fig, run_path, y_label, y_pred, name):

    conf_mat = confusion_matrix(y_label, y_pred)
    print(name + "_Confusion_Matrix: \n", conf_mat, "\n")
    file.write(name + "_Confusion Matrix: \n" + str(conf_mat) + "\n\n")

    fig = plt.figure(figsize=(10, 10))
    plt.title(name + '_Confusion_matrix')
    ax = sns.heatmap(conf_mat, annot = True, cmap = plt.cm.Blues, fmt = 'g', linewidths = 1)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set(ylabel = "True Labels", xlabel = "Predicted Labels")

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if(save_fig):
        save_path = os.path.join(run_path, name + "_Confusion_Matrix.png")
        fig.savefig(save_path, bbox_inches = 'tight')
    fig.clf()
    plt.close(fig)

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
    for i in range(len(target_names)):
        print(target_names[i], ": ", round(class_accuracy[i],2), "%")
        file.write(target_names[i] + ":" + str(round(class_accuracy[i],2)) + "% \n")
    print()
    file.write("\n")



#---------------------------------------------------------------------
# Function:    visual_tsne()
# Description: Visualize TSNE
#---------------------------------------------------------------------
def visual_tsne(file, save_data, save_fig, run_path, run_image_path, features, preds, labels, class_id_dict, name):

    # Color for each class (used in scatter plot)
    colors_dict = {1 : 'tab:blue'  ,
                   2 : 'tab:orange',
                   5 : 'tab:green' ,
                   6 : 'tab:red'   ,
                   9 : 'tab:purple',
                   11: 'tab:brown' ,
                   12: 'tab:gray'  ,
                   13: 'tab:pink'  ,
                   14: 'tab:cyan'  }

    # Constants
    N       = 2
    PERP    = [3000.0]
    LR      = 'auto'
    ITER    = [5000]
    VERBOSE = 1
    STATE   = 0

    for i in PERP:
        for j in ITER:
            print("n_components = {}, perplexity = {}, learning_rate = {}, n_iter = {}, verbose = {}, random_state = {}".format(N, i, LR, j, VERBOSE, STATE))
            file.write("n_components = {}, perplexity = {}, learning_rate = {}, n_iter = {}, verbose = {}, random_state = {} \n".format(N, i, LR, j, VERBOSE, STATE))
            tsne = TSNE(n_components = N, perplexity = i,  learning_rate = LR, n_iter = j,  verbose = VERBOSE, random_state = STATE).fit_transform(features)

            # print("tsne:", tsne)
            # print("tsne.shape:", tsne.shape)

            if(save_data):
                tsne_data = pd.DataFrame(tsne)
                tsne_data.columns = ["x_coord", "y_coord"]
                tsne_data["labels"] = labels
                tsne_data.to_csv(os.path.join(run_path, name + "_" + str(i) + "_" + str(j) + "_tsne_data.csv"))

            # Extract x and y coordinates representing the positions of the images on T-SNE plot
            tsne_x = tsne[:, 0]
            tsne_y = tsne[:, 1]

            # Plot all labels
            fig_all = plt.figure(num = 0, figsize = (10, 10))
            plt.title(name + '_All_TSNE_Labeled')
            plt.xlabel('t-SNE-1')
            plt.ylabel('t-SNE-2')

            indices = []
            for key, value in class_id_dict.items():

                # Plot each label
                fig = plt.figure(num = 1, figsize = (10, 10))
                plt.title(name + '_' + str(value) + '_TSNE_Labeled')
                plt.xlabel('t-SNE-1')
                plt.ylabel('t-SNE-2')

                # Find all indices for each class
                for i, pred in enumerate(preds):
                    if(pred == key):
                        indices.append(i)

                # Extract the coordinates of the points of this class only
                current_tsne_x = np.take(tsne_x, indices)
                current_tsne_y = np.take(tsne_y, indices)

                fig_all = plt.figure(num = 0, figsize = (10, 10))
                plt.scatter(current_tsne_x, current_tsne_y, s = 2, c = colors_dict.get(key), label = value)
                plt.legend(loc='best')

                fig = plt.figure(num = 1, figsize = (10, 10))
                plt.scatter(current_tsne_x, current_tsne_y, s = 2, c = colors_dict.get(key), label = value)
                plt.legend(loc='best')

                if(save_fig):
                    save_path = os.path.join(run_image_path, name + "_" + str(i) + "_" + str(j) + "_" + value + "_TSNE_Labeled.png")
                    fig.savefig(save_path, bbox_inches = 'tight')
                fig.clf()
                plt.close(fig)
                indices.clear() # Clear indices list

            if(save_fig):
                plt.legend(loc='best')
                save_path = os.path.join(run_image_path, name + "_" + str(i) + "_" + str(j) + "_All_TSNE_Labeled.png")
                fig_all.savefig(save_path, bbox_inches = 'tight')
            fig_all.clf()
            plt.close(fig_all)

        file.write("\n")



#---------------------------------------------------------------------
# Function:    test_encoder_decoder()
# Description: Create syntheic images using Encoder Decoder and test using vgg16 model
#---------------------------------------------------------------------
def helper_test_encoder_decoder(args, file, run_path, number_classes, test_csv_file, vgg_model_path, ec_model_path_params, ec_model_path_zeroparams, df_test_encoder_decoder):

    run_images_path = os.path.join(run_path, "run_images")
    os.mkdir(run_images_path)

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
    # vgg_model.classifier[6] = nn.Linear(num_ftrs, num_classes + 6) # Classes: 1, 2, 5, 6, 9, 11, 12, 13, 14 [Total: 15]
    vgg_model.load_state_dict(torch.load(vgg_model_path))
    vgg_model = vgg_model.to(device)

    # Model - Encoder-Decoder
    ec_model_params = model_encoder_decoder()
    ec_model_params = ec_model_params.to(device)
    ec_model_params.load_state_dict(torch.load(ec_model_path_params))
    ec_model_params = ec_model_params.to(device)

    ec_model_zeroparams = model_encoder_decoder()
    ec_model_zeroparams = ec_model_zeroparams.to(device)
    ec_model_zeroparams.load_state_dict(torch.load(ec_model_path_zeroparams))
    ec_model_zeroparams = ec_model_zeroparams.to(device)

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

        zero_param.append([0,0,0,0,0])

    param      = torch.FloatTensor(param)
    zero_param = torch.FloatTensor(zero_param)

    # print(df_test_encoder_decoder.info())
    channel = list(input_size)
    channel.insert(0, 3)      # (3, 64, 64)
    channel = tuple(channel)
    flip_channel = list(input_size)
    flip_channel.insert(2, 3) # (64, 64, 3)
    flip_channel = tuple(flip_channel)

    testing_set            = Encoder_Decoder_Dataset(df_test_encoder_decoder, input_size, channel)
    test_loader            = DataLoader(testing_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    param_test_loader      = DataLoader(param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)
    zero_param_test_loader = DataLoader(zero_param, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)

    # Test Model
    vgg_model.eval()
    ec_model_params.eval()
    ec_model_zeroparams.eval()

    # Ground truth of image (without going through encoder-decoder, just classifier)
    GT_label  = torch.zeros(0, dtype = torch.long, device = 'cpu')
    GT_pred   = torch.zeros(0, dtype = torch.long, device = 'cpu')
    GT_output = torch.zeros(0, dtype = torch.long, device = 'cpu')

    # Synthic view of image 2 with zero params (aka ground truth) (goes through EC and classifier)
    synth_GT_label  = torch.zeros(0, dtype = torch.long, device = 'cpu')
    synth_GT_pred   = torch.zeros(0, dtype = torch.long, device = 'cpu')
    synth_GT_output = torch.zeros(0, dtype = torch.long, device = 'cpu')

    # Synthic view of image 1 with params (aka new image 2) (goes through EC and classifier)
    synth_new_image_label  = torch.zeros(0, dtype = torch.long, device = 'cpu')
    synth_new_image_pred   = torch.zeros(0, dtype = torch.long, device = 'cpu')
    synth_new_image_output = torch.zeros(0, dtype = torch.long, device = 'cpu')

    # Ground truth of image (without going through encoder-decoder or classifier)
    # original = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        params_iter      = iter(param_test_loader)
        zero_params_iter = iter(zero_param_test_loader)

        for batch, (images, labels) in enumerate(test_loader):

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

            # Classify images & append batch prediction results
            vgg_GT_outputs  = vgg_model(images_2)
            _, vgg_GT_preds = torch.max(vgg_GT_outputs, 1)
            GT_label        = torch.cat([GT_label,  labels_2.view(-1).cpu()])
            GT_pred         = torch.cat([GT_pred,   vgg_GT_preds.view(-1).cpu()])
            GT_output       = torch.cat([GT_output, vgg_GT_outputs.cpu()])

            # Create synthic view of image 2 with zero params (aka ground truth)
            ec_GT_outputs      = ec_model_zeroparams(images_2, zero_params)
            vgg_ec_GT_outputs  = vgg_model(ec_GT_outputs)
            _, vgg_ec_GT_preds = torch.max(vgg_ec_GT_outputs, 1)
            synth_GT_label     = torch.cat([synth_GT_label,  labels_2.view(-1).cpu()])
            synth_GT_pred      = torch.cat([synth_GT_pred,   vgg_ec_GT_preds.view(-1).cpu()])
            synth_GT_output    = torch.cat([synth_GT_output, vgg_ec_GT_outputs.cpu()])

            # Create Synthic view of image 1 with params (aka new image 2)
            ec_new_image_outputs      = ec_model_params(images_1, params)
            vgg_ec_new_image_outputs  = vgg_model(ec_new_image_outputs)
            _, vgg_ec_new_image_preds = torch.max(vgg_ec_new_image_outputs, 1)
            synth_new_image_label     = torch.cat([synth_new_image_label,  labels_2.view(-1).cpu()])
            synth_new_image_pred      = torch.cat([synth_new_image_pred,   vgg_ec_new_image_preds.view(-1).cpu()])
            synth_new_image_output    = torch.cat([synth_new_image_output, vgg_ec_new_image_outputs.cpu()])

            # original           = torch.cat([original, images_2.view(images_2.size(0),-1).cpu()])

            # Sample Images
            if (batch == 0) or (batch % 21 == 0):
                print("*" * 10)
                file.write("*" * 10 +"\n")
                print("BATCH {}:".format(batch))
                file.write("BATCH {}:".format(batch) + "\n")

                fig = plt.figure(figsize=(20, 20))
                fig.add_subplot(1, 4, 1)
                plt.title("images_1_GT[0]")
                a = images_1[0] * 65535                 # De-normalize
                a = a.reshape((flip_channel))           # Reshape (3, 64, 64) -> (64, 64, 3)
                a = a.detach().to("cpu").numpy()        # Covert to "CPU" and numpy array
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) # Remove 3 channels
                plt.imshow(a, cmap = 'gray')            # Display on color map
                # print(images_1[0], type(images_1[0]), images_1[0].size())
                # print(a, type(a), a.shape)
                # print(type(a))
                # print(a.shape)
                # print()

                fig.add_subplot(1, 4, 2)
                plt.title("images_2_GT[0]")
                b = images_2[0] * 65535
                b = b.reshape((flip_channel))
                b = b.detach().to("cpu").numpy()
                b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                plt.imshow(b, cmap = 'gray')
                # print(images_2[0], type(images_2[0]), images_2[0].size())
                # print(type(b))
                # print(b.shape)
                # print()

                fig.add_subplot(1, 4, 3)
                plt.title("synth_image2_GT[0]")
                c = ec_GT_outputs[0] * 65535
                c = c.reshape((flip_channel))
                c = c.detach().to("cpu").numpy()
                c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                plt.imshow(c, cmap = 'gray')
                # print(ec_GT_outputs[0], type(ec_GT_outputs[0]), ec_GT_outputs[0].size())
                # print(type(c))
                # print(c.shape)
                # print()

                fig.add_subplot(1, 4, 4)
                plt.title("synth_image2_new_image[0]")
                d = ec_new_image_outputs[0] * 65535
                d = d.reshape((flip_channel))
                d = d.detach().to("cpu").numpy()
                d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                plt.imshow(d, cmap = 'gray')
                # print(ec_new_image_outputs[0], type(ec_new_image_outputs[0]), ec_new_image_outputs[0].size())
                # print(type(d))
                # print(d.shape)
                # print()

                if (save_fig):
                    save_path = os.path.join(run_images_path, (str(batch) + "_ run_images.png"))
                    plt.savefig(save_path, bbox_inches = 'tight')

                fig.clf()
                plt.close(fig)

    print("*" * 10 + "\n")
    file.write("*" * 10 +"\n\n")

    GT_label  = GT_label.numpy()
    GT_pred   = GT_pred.numpy()
    GT_output = GT_output.numpy()

    synth_GT_label  = synth_GT_label.numpy()
    synth_GT_pred   = synth_GT_pred.numpy()
    synth_GT_output = synth_GT_output.numpy()

    synth_new_image_label  = synth_new_image_label.numpy()
    synth_new_image_pred   = synth_new_image_pred.numpy()
    synth_new_image_output = synth_new_image_output.numpy()

    # original     = original.numpy()

    # Print count of each label
    for key in class_id_dict.keys():
        print(key, " count(GT_pred): ",              np.count_nonzero(GT_pred == key),
                   " count(synth_GT_pred): ",        np.count_nonzero(synth_GT_pred == key),
                   " count(synth_new_image_pred): ", np.count_nonzero(synth_new_image_pred == key))

        file.write(str(key) + " count(GT_pred): " +              str(np.count_nonzero(GT_pred == key))
                            + " count(synth_GT_pred): " +        str(np.count_nonzero(synth_GT_pred == key))
                            + " count(synth_new_image_pred): " + str(np.count_nonzero(synth_new_image_pred == key)) + "\n")
    print()
    file.write("\n")

    if(save_data):
        df_test_csv            = pd.read_csv(test_csv_file)
        test_image             = df_test_csv['image_2']
        GT_length              = len(GT_label)
        synth_GT_length        = len(synth_GT_label)
        synth_new_image_length = len(synth_new_image_label)

        with open(os.path.join(run_path, "test_classifier_GT_data.csv"), 'w', newline ='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(["image", "ground_truth", "predict"])
            for i in range(GT_length):
                write.writerow([test_image[i], class_id_dict[GT_label[i]], class_id_dict[GT_pred[i]]])

        with open(os.path.join(run_path, "test_encoder_decoder_classifier_GT_data.csv"), 'w', newline ='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(["image", "ground_truth", "predict"])
            for i in range(synth_GT_length):
                write.writerow([test_image[i], class_id_dict[synth_GT_label[i]], class_id_dict[synth_GT_pred[i]]])

        with open(os.path.join(run_path, "test_encoder_decoder_classifier_new_image_data.csv"), 'w', newline ='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(["image", "ground_truth", "predict"])
            for i in range(synth_new_image_length):
                write.writerow([test_image[i], class_id_dict[synth_new_image_label[i]], class_id_dict[synth_new_image_pred[i]]])

    target_names = ['PICKUP','SUV','BTR70','BRDM2','BMP2','T72','ZSU23', '2S3', 'D20']

    # Accuracy Score
    GT_acc_score = accuracy_score(GT_label, GT_pred)
    print("Ground Truth Model Accuracy: ", GT_acc_score)
    file.write("Ground Truth Model Accuracy: " + str(GT_acc_score) + "\n")

    synth_GT_acc_score = accuracy_score(synth_GT_label, synth_GT_pred)
    print("Synthetic GT Model Accuracy: ", synth_GT_acc_score)
    file.write("Synthetic GT Model Accuracy: " + str(synth_GT_acc_score) + "\n")

    synth_new_image_acc_score = accuracy_score(synth_new_image_label, synth_new_image_pred)
    print("Synthetic New Image Model Accuracy: ", synth_new_image_acc_score, "\n")
    file.write("Synthetic New Image Model Accuracy: " + str(synth_new_image_acc_score) + "\n\n")

    # Confusion Matrix
    plot_confusion_matrix(file, target_names, save_fig, run_path, GT_label, GT_pred, "Ground_Truth")
    plot_confusion_matrix(file, target_names, save_fig, run_path, synth_GT_label, synth_GT_pred, "Encoder_Decoder_Old_View")
    plot_confusion_matrix(file, target_names, save_fig, run_path, synth_new_image_label, synth_new_image_pred, "Encoder_Decoder_New_View")

    # T-distributed Stochastic Neighbor Embedding
    # visual_tsne(file, save_fig, run_path, run_images_path, original, truth_label, class_id_dict, "Original")
    # visual_tsne(file, save_data, save_fig, run_path, run_images_path, GT_output, GT_pred, GT_label, class_id_dict, "Ground_Truth")
    visual_tsne(file, save_data, save_fig, run_path, run_images_path, synth_GT_output, synth_GT_pred, synth_GT_label, class_id_dict, "Encoder_Decoder_Old_View")
    visual_tsne(file, save_data, save_fig, run_path, run_images_path, synth_new_image_output, synth_new_image_pred, synth_new_image_label, class_id_dict, "Encoder_Decoder_New_View")

#=====================================================================