#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      main.py
# Create:        01/17/22
# Description:   Main function to call other functions
#---------------------------------------------------------------------

# IMPORTS:
# From Functions Directory
from Functions.process_classifier_data import *
from Functions.process_encoder_decoder_data import *
from Functions.test_classifier import *
from Functions.test_encoder_decoder import *
from Functions.train_classifier import *
from Functions.train_encoder_decoder import *

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():

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
    classifier_data = "Input/Train_Classifier_Dataset"

    # Processing Step
    process_classifier_data(classifier_data)
    process_encoder_decoder_data()

    # Training Step
    train_classifier()
    train_encoder_decoder()

    # Evaluation Step
    test_classifier()
    test_encoder_decoder()

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================