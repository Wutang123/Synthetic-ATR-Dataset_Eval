#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_encoder_decoder_images2csv.py
# Create:        02/25/22
# Description:   Read directory of images and convert to csv format
#                Used for training encoder decoder
#---------------------------------------------------------------------

# IMPORTS:
import os
import csv

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    train_encoder_decoder_images2csv()
# Description: Read directory of images and convert to csv format
#---------------------------------------------------------------------
def train_encoder_decoder_images2csv(input_path, dir_path):

    csv_path = os.path.join(input_path, "Train_Encoder_Decoder_Dataset.csv")
    print("Output CSV Path:     ", csv_path)

    header = ['image_1', 'class_id_1', 'class_type_1', 'time_of_day_1', 'range_1', 'orientation_1',
                'image_2', 'class_id_2', 'class_type_2', 'time_of_day_2', 'range_2', 'orientation_2']
    data = []

    train_list = []
    # Only capture frames every 10 degrees (36 images per class)
    # Roughly every 25 images between frame 901 to 1800
    for i in range(901, 1800, 25):
        train_list.append(i)
    # print(train_list)

    with open(csv_path, 'w', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write Header
        writer.writerow(header)

        for sub_dir_name in os.listdir(dir_path):
            sub_dir_path = os.path.join(dir_path, sub_dir_name)
            print(sub_dir_path)

            for file_name_1 in os.listdir(sub_dir_path):
                file_path_1 = os.path.join(sub_dir_path, file_name_1)

                # Check if file (file_path_1)
                if os.path.isfile(file_path_1) and (not file_path_1.endswith('.ini')):
                    # print("1",file_path_1)
                    substring_1 = file_name_1.split('_')

                    if(int(substring_1[2]) in train_list):
                        for file_name_2 in os.listdir(sub_dir_path):
                            file_path_2 = os.path.join(sub_dir_path, file_name_2)

                            # Check if file (file_path_2)
                            if os.path.isfile(file_path_2) and (not file_path_2.endswith('.ini')):
                                # print("2",file_path_2)
                                substring_2 = file_name_2.split('_')

                                if(int(substring_2[2]) in train_list):
                                    if(substring_1[3].replace(".mat","") == substring_2[3].replace(".mat","")):

                                        # First File
                                        data.append(file_path_1)                       # Data[0]
                                        data.append(substring_1[1])                    # Data[1]
                                        data.append(substring_1[3].replace(".mat","")) # Data[2]
                                        if substring_1[0] == 'cegr01923':
                                            data.append("Night")                       # Data[3]
                                            data.append(1000)                          # Data[4]
                                        elif substring_1[0] == 'cegr01927':
                                            data.append("Night")
                                            data.append(2000)
                                        elif substring_1[0] == 'cegr02003':
                                            data.append("Day")
                                            data.append(1000)
                                        elif substring_1[0] == 'cegr02007':
                                            data.append("Day")
                                            data.append(2000)
                                        data.append(substring_1[2])                    # Data[5]

                                        # Second File
                                        data.append(file_path_2)                       # Data[6]
                                        data.append(substring_2[1])                    # Data[7]
                                        data.append(substring_2[3].replace(".mat","")) # Data[8]
                                        if substring_2[0] == 'cegr01923':
                                            data.append("Night")                       # Data[9]
                                            data.append(1000)                          # Data[10]
                                        elif substring_2[0] == 'cegr01927':
                                            data.append("Night")
                                            data.append(2000)
                                        elif substring_2[0] == 'cegr02003':
                                            data.append("Day")
                                            data.append(1000)
                                        elif substring_2[0] == 'cegr02007':
                                            data.append("Day")
                                            data.append(2000)
                                        data.append(substring_2[2])                    # Data[11]

                                        # print(data, "\n")

                                        # Write Data
                                        writer.writerow(data)
                                        data.clear()
#=====================================================================