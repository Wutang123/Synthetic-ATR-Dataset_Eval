#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      train_encoder_decoder.py
# Create:        01/17/22
# Description:   Main train encoder decoder function to call helper functions
#---------------------------------------------------------------------

# IMPORTS:
import os
from datetime import datetime
import argparse
import math
import time
import pandas as pd
from sklearn.utils import shuffle

# From Functions Directory
from Helper_Functions.helper_train_encoder_decoder import *

# Constant
SHUFFLE_SEED = 0

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    main()
# Description: Main Functions; calls other functions
#---------------------------------------------------------------------
def main():
    startT = time.time()
    parser = argparse.ArgumentParser(description = 'train_encoder_decoder')
    parser.add_argument('--save_fig'  , type = bool , default = True                                       , help = 'Save Figures')
    parser.add_argument('--save_model', type = bool , default = True                                       , help = 'Save Model')
    parser.add_argument('--save_data' , type = bool , default = True                                       , help = 'Save Data to CSV')
    parser.add_argument('--csv'       , type = str  , default = 'Input/Train_Encoder_Decoder_Dataset.csv'  , help = 'Load csv files')
    parser.add_argument('--lr'        , type = float, default = 1e-4                                       , help = 'Select Learning Rate (e.g. 1e-4)')
    parser.add_argument('--epoch'     , type = int  , default = 10                                         , help = 'Select Epoch Size (e.g 25)')
    parser.add_argument('--batch'     , type = int  , default = 64                                         , help = 'Select Batch Size (e.g 64)')
    parser.add_argument('--worker'    , type = int  , default = 2                                          , help = 'Select Number of Workers (e.g 4)')
    parser.add_argument('--imgsz'     , type = int  , default = (64,64)                                    , help = 'Select Input Image Size (e.g 64,64)')
    args = parser.parse_args()

    print(">>>>> train_encoder_decoder \n")
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H.%M.%S")

    # Create Output file
    cont = True
    count = 0
    run_path = os.path.join("OUTPUT", "train_encoder_decoder", "Run" + str(count))
    while cont:
        if(os.path.isdir(run_path)):
            count += 1
            run_path = os.path.join("OUTPUT", "train_encoder_decoder", "Run" + str(count))
        else:
            os.mkdir(run_path)
            cont = False

    # Create Logfile
    log_file = os.path.join(run_path, "log_file.txt")
    file = open(log_file, "a")
    file.write("=" * 10 + "\n")
    file.write("Log File Generated On: "+ date_time + "\n")
    file.write("-" * 10 + "\n")
    print(args, "\n")
    file.write(str(args) + "\n")

    print("Input csv file: ", args.csv, "\n")
    file.write("Input csv file: " + str(args.csv) + "\n\n")

    df_train_encoder_decoder = pd.read_csv(args.csv, index_col = 0)
    df_train_encoder_decoder = df_train_encoder_decoder.reset_index()
    df_train_encoder_decoder = shuffle(df_train_encoder_decoder, random_state = SHUFFLE_SEED)
    df_train_encoder_decoder = df_train_encoder_decoder.reset_index(drop = True)
    # print(df_train_encoder_decoder)

    number_classes = 9
    helper_train_encoder_decoder(args, file, run_path, number_classes, df_train_encoder_decoder)

    endT = time.time()
    program_time_difference = endT - startT
    min = math.floor(program_time_difference / 60)
    sec = math.floor(program_time_difference % 60)
    print("Total Program Time (min:sec): " + str(min) + ":" + str(sec))
    file.write("Total Program Time (min:sec): " + str(min) + ":" + str(sec) + "\n")

    file.write("=" * 10)
    file.close()

# MODULES:
if __name__ == "__main__":
    main()

#=====================================================================