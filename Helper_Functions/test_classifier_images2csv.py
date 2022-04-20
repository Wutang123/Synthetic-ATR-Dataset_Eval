#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      test_classifier_images2csv.py
# Create:        02/25/22
# Description:   Read directory of images and convert to csv format
#                Used for testing classifier
#---------------------------------------------------------------------

# IMPORTS:
import os
import csv

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    test_classifier_images2csv()
# Description: Read directory of images and convert to csv format
#---------------------------------------------------------------------
def test_classifier_images2csv(input_path, dir_path):

    csv_path = os.path.join(input_path, "Test_Classifier_Dataset.csv")
    print("Output CSV Path:     ", csv_path)

    header = ['image', 'class_id', 'class_type', 'time_of_day', 'range', 'orientation']
    data = []

    test_list = []
    # Only capture frames every 10 degrees (36 images per class)
    # Roughly every 25 images between frame 1 to 900
    for i in range(1, 900, 25):
        test_list.append(i)
    # print(test_list)

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

                    if(int(substring[2]) in test_list):
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