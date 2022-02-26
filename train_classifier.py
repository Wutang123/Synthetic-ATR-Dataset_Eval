#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_classifier.py
# Create:        01/17/22
# Description:   Main train classifier function to call helper functions
#---------------------------------------------------------------------

# IMPORTS:
import os
from datetime import datetime
import argparse
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
    parser = argparse.ArgumentParser(description = 'train_classifier')
    # parser.add_argument('--analysis'  , type = bool , default = False, help = 'Conduct EDA')
    # parser.add_argument('--save_fig'  , type = bool , default = True , help = 'Save Figures')
    # parser.add_argument('--save_model', type = bool , default = True , help = 'Save Model')
    # parser.add_argument('--save_data' , type = bool , default = True , help = 'Save Data to CSV')
    # parser.add_argument('--pretrained', type = bool , default = True , help = 'Use Pretrained Models (e.g True)')
    # parser.add_argument('--lr'        , type = float, default = 1e-4 , help = 'Select Learning Rate (e.g. 1e-3)')
    # parser.add_argument('--epoch'     , type = int  , default = 50   , help = 'Select Epoch Size (e.g 50)')
    # parser.add_argument('--batch'     , type = int  , default = 32   , help = 'Select Batch Size (e.g 32)')
    # parser.add_argument('--worker'    , type = int  , default = 4    , help = 'Select Number of Workers (e.g 4)')
    # parser.add_argument('--imgsz'     , type = int  , default = 225  , help = 'Select Input Image Size (e.g 225)')
    args = parser.parse_args()

    # TODO:
    # Add logfile feature

    print(">>>>> train_classifier \n")

    cont = True
    count = 0
    run_path = os.path.join("OUTPUT", "train_classifier", "Run" + str(count))
    while cont:
        if(os.path.isdir(run_path)):
            count += 1
            run_path = os.path.join("OUTPUT", "train_classifier", "Run" + str(count))
        else:
            os.mkdir(run_path)
            cont = False

    # Load Training and Testing Data via csv file
    train_classifier_csv      = "Input/Train_Classifier_Dataset.csv"

    # df_train_classifier = pd.read_csv(train_classifier_csv, index_col = 0)
    # df_train_classifier = df_train_classifier.reset_index()
    # train_classifier(run_path, df_train_classifier)

    endT = time.time()
    program_time_difference = endT - startT
    min = math.floor(program_time_difference/60)
    sec = math.floor(program_time_difference%60)
    print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================