#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      images2csv.py
# Create:        02/25/22
# Description:   Main function to create csv files
#---------------------------------------------------------------------

# IMPORTS:
import time
import argparse
import math

# From Functions Directory
from Helper_Functions.train_classifier_images2csv import *
from Helper_Functions.test_classifier_images2csv import *
from Helper_Functions.train_encoder_decoder_images2csv import *
from Helper_Functions.test_encoder_decoder_images2csv import *

startT = time.time()
parser = argparse.ArgumentParser(description = 'images2csv')
parser.add_argument('--train_classifier'       , type = bool , default = False, help = 'Create CSV file for training classifier (True or False)')
parser.add_argument('--test_classifier'        , type = bool , default = False, help = 'Create CSV file for testing classifier (True or False)')
parser.add_argument('--train_encoder_decoder'  , type = bool , default = False, help = 'Create CSV file for training encoder_decoder (True or False)')
parser.add_argument('--test_encoder_decoder'   , type = bool , default = True, help = 'Create CSV file for testing encoder_decoder (True or False)')
args = parser.parse_args()

print(">>>>> images2csv \n")

input_path = "INPUT"
dir_path = os.path.join(input_path, "Dataset")
print("Input Dataset Path: ", dir_path)

if(args.train_classifier):
    train_classifier_images2csv(input_path, dir_path)
    print()

if(args.test_classifier):
    test_classifier_images2csv(input_path, dir_path)
    print()

if(args.train_encoder_decoder):
    train_encoder_decoder_images2csv(input_path, dir_path)
    print()

if(args.test_encoder_decoder):
    test_encoder_decoder_images2csv(input_path, dir_path)
    print()

endT = time.time()
program_time_difference = endT - startT
min = math.floor(program_time_difference / 60)
sec = math.floor(program_time_difference % 60)
print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))

#=====================================================================