#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_classifier_images2csv.py
# Create:        02/25/22
# Description:   Read directory of images and convert to csv format
#                Used for training classifier
#---------------------------------------------------------------------

# IMPORTS:
import os
import csv

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train_classifier_images2csv()
# Description: Read directory of images and convert to csv format
#---------------------------------------------------------------------
def train_classifier_images2csv(input_path, dir_path):

    csv_path = os.path.join(input_path, "Train_Classifier_Dataset.csv")
    print("Output CSV Path:     ", csv_path)

    header = ['image', 'class_id', 'class_type', 'time_of_day', 'range', 'orientation']
    data = []

    with open(csv_path, 'w', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write Header
        writer.writerow(header)

        for sub_dir_name in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_dir_name)
            print(sub_dir_path)

            for file_name in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file_name)

                # Check if file
                if os.path.isfile(file_path) and (not file_path.endswith('.ini')):
                    # print(file_path)
                    substring = file_name.split('_')

                    if(int(substring[2]) > 900):
                        data.append(file_path)                       # Data[0]
                        data.append(substring[1])                    # Data[1]
                        data.append(substring[3].replace(".mat","")) # Data[2]
                        if substring[0] == 'cegr01923':
                            data.append("Night")                     # Data[3]
                            data.append(1000)                        # Data[4]
                        elif substring[0] == 'cegr01927':
                            data.append("Night")
                            data.append(2000)
                        elif substring[0] == 'cegr02003':
                            data.append("Day")
                            data.append(1000)
                        elif substring[0] == 'cegr02007':
                            data.append("Day")
                            data.append(2000)
                        data.append(substring[2])                    # Data[5]

                        # print(data, "\n")

                        # Write Data
                        writer.writerow(data)
                        data.clear()
#=====================================================================