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
import seaborn as sns
import scipy.io # Open .mat files
import cv2

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
# Function:    test_classifier()
# Description: Evaluate Classifier
#---------------------------------------------------------------------
def test_classifier(run_path, trained_classifier_path, df_test_classifier):
    # TODO: Update Hyperparameters Later
    batch =  64
    num_worker = 4
    input_size = (80,40) # Minimum for VGG16 is (32,32)
    num_classes = 9

    # Define the device (use GPU if avaliable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device Used:       ", device, "\n")

    # Model
    model = models.vgg16(pretrained = False)

    # Update number of input channels from 3 to 1
    first_conv_layer = [nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)]
    first_conv_layer.extend(list(model.features))
    model.features= nn.Sequential(*first_conv_layer)

    # Update number of output classes from 1000 to 9
    num_ftrs = model.classifier[6].in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device) # Add device to model

    # Load Model
    model.load_state_dict(torch.load(trained_classifier_path))

    test_set    = Dataset(df_test_classifier, input_size)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, num_workers = num_worker, drop_last = True)

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

    target_names = ['PICKUP','SUV','BTR70','BRDM2','BMP2','T72','ZSU23', '2S3', 'D20']

    # Accuracy Score
    acc_score = accuracy_score(y_label, y_pred)
    print(acc_score)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_label, y_pred)
    print("Confusion Matrix: \n", conf_mat, "\n")

    fig = plt.figure()
    plt.title('Confusion matrix')
    ax = sns.heatmap(conf_mat, annot = True, cmap = plt.cm.Blues, fmt = 'g', linewidths = 1)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set(ylabel = "True Labels", xlabel = "Predicted Labels")

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    save_path = os.path.join(run_path, "Confusion_Matrix.png")

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
    for i in range(len(target_names)):
        print(target_names[i], ": ", round(class_accuracy[i],2), "%")

#=====================================================================