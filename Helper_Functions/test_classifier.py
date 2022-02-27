#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      test_classifier.py
# Create:        01/17/22
# Description:   Evaluate Classifier Performance
#---------------------------------------------------------------------

# IMPORTS:
import os
import matplotlib.pyplot as plt
from numpy import save
import seaborn as sns
import scipy.io # Open .mat files
import cv2
import csv
import pandas as pd
import json

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchsummary import summary

# sklearn libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Class:
#---------------------------------------------------------------------
# Function:    Dataset()
# Description: Characterizes ATR Dataset for PyTorch
#---------------------------------------------------------------------
class Dataset(Dataset):
    def __init__(self, df, resize):
        self.images = df['image']
        self.ids = df['class_id']
        self.resize = resize

    def __getitem__(self, index):
        image = scipy.io.loadmat(self.images[index], squeeze_me = True)
        # keys = x.keys()
        # print(keys)
        # dict_keys(['__header__', '__version__', '__globals__', 'target_chip'])
        image = image['target_chip']
        label = torch.tensor(int(self.ids[index]))

        if self.resize:
            image = cv2.resize(image, self.resize)
            image = torch.from_numpy(image).float()

        return image, label

    def __len__(self):
        return len(self.images)


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
# Function:    test_classifier()
# Description: Evaluate Classifier
#---------------------------------------------------------------------
def test_classifier(args, file, run_path, number_classes, test_csv_file, model_path, df_test_classifier):
    batch       = args.batch
    num_worker  = args.worker
    input_size  = args.imgsz # Minimum for VGG16 is (32,32)
    num_classes = number_classes
    save_fig    = args.save_fig
    save_data   = args.save_data

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

    class_id_dict = {
                     1 : 'PICKUP',
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
    model = models.vgg16(pretrained = False)

    # Update number of input channels from 3 to 1
    first_conv_layer = [nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)]
    first_conv_layer.extend(list(model.features))
    model.features= nn.Sequential(*first_conv_layer)

    # Update number of output classes from 1000 to 9
    num_ftrs = model.classifier[6].in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load Trained Model
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) # Add device to model

    test_set    = Dataset(df_test_classifier, input_size)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = False)

    # Test Model
    model.eval()
    y_label    = torch.zeros(0, dtype = torch.long, device = 'cpu')
    y_pred     = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        for images, labels in test_loader:
            images = torch.unsqueeze(images, dim = 1) # Add channel dimension = 1

            images   = images.to(device)
            labels   = labels.to(device)
            outputs  = model(images)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            y_label    = torch.cat([y_label, labels.view(-1).cpu()])
            y_pred     = torch.cat([y_pred, preds.view(-1).cpu()])

    y_label = y_label.numpy()
    y_pred = y_pred.numpy()

    if(save_data):
        df_test_csv = pd.read_csv(test_csv_file)
        test_image = df_test_csv['image']

        # Open csv file in 'w' mode
        with open(os.path.join(run_path, "test_classifier_data.csv"), 'w', newline ='') as csv_file:
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