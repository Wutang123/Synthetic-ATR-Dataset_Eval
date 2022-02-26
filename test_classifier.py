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
# Description:   Main test classifier function to call helper functions
#---------------------------------------------------------------------

# IMPORTS:
import os
from datetime import datetime
import math
import pandas as pd

# From Functions Directory
from Helper_Functions.test_classifier import *
from Helper_Functions.test_encoder_decoder import *
from Helper_Functions.train_classifier import *
from Helper_Functions.train_encoder_decoder import *


# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    startT = time.time()

    # TODO:
    # Add logfile feature
    # Add arg parse
        # Read in classifier dataset
        # Read in encoder decoder dataset
        # Read in if classifer has been trained already
        # Read in if encoder decoder has been trained already
        # Save model
        # Save Data
        # Take in hyperparameters

    print(">>>>> Starting Program")

    cont = True
    count = 0
    run_path = os.path.join("OUTPUT\Run" + str(count))
    while cont:
        if(os.path.isdir(run_path)):
            count += 1
            run_path = os.path.join("OUTPUT\Run" + str(count))
        else:
            os.mkdir(run_path)
            cont = False

    # TODO: Change later using args
    train_classifier_model =      False
    train_encoder_decoder_model = False
    test_classifier_model =       True
    test_encoder_decoder_model =  False

    # Load Training and Testing Data via csv file
    train_classifier_csv      = "Input/Train_Classifier_Dataset.csv"
    train_encoder_decoder_csv = "Input/Train_Encoder_Decoder_Dataset.csv"
    test_classifier_csv       = "Input/Test_Classifier_Dataset.csv"
    test_encoder_decoder_csv  = "Input/Test_Encoder_Decoder_Dataset.csv"

    df_train_classifier = pd.read_csv(train_classifier_csv, index_col = 0)
    df_train_classifier = df_train_classifier.reset_index()
    df_train_encoder_decoder = pd.read_csv(train_encoder_decoder_csv, index_col = 0)
    df_train_encoder_decoder = df_train_encoder_decoder.reset_index()

    df_test_classifier = pd.read_csv(test_classifier_csv, index_col = 0)
    df_test_classifier = df_test_classifier.reset_index()
    df_test_encoder_decoder = pd.read_csv(test_encoder_decoder_csv, index_col = 0)
    df_test_encoder_decoder = df_test_encoder_decoder.reset_index()

    # Load Model
    trained_classifier_path = os.path.join("OUTPUT", "Run0","classifier.pth") # TODO: Remove later
    trained_encoder_decoder_path = os.path.join("OUTPUT", "Run0","encoder_decoder.pth") # TODO: Remove later

    # Training Step
    if(train_classifier_model):
        trained_classifier_path = train_classifier(run_path, df_train_classifier)
    if(train_encoder_decoder_model):
        trained_encoder_decoder_path = train_encoder_decoder(run_path, df_train_encoder_decoder)

    # Evaluation Step
    if(test_classifier_model):
        test_classifier(run_path, trained_classifier_path, df_test_classifier)
    if(test_encoder_decoder_model):
        test_encoder_decoder(run_path, trained_encoder_decoder_path, df_test_encoder_decoder)

    print(">>>>> Ending Program")
    endT = time.time()
    program_time_difference = endT - startT
    min = math.floor(program_time_difference/60)
    sec = math.floor(program_time_difference%60)
    print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================