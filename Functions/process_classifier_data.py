#=====================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------
# Class:         CAP6908 - Independent Studies
# Professor:     Professor Abhijit Mahalanobis
# Name:          Justin Wu
# Project:       ATR Dataset Evaluator
# Function:      process_classifier_data.py
# Create:        01/17/22
# Description:   Process Train_Classifier_Dataset for training and testing
#---------------------------------------------------------------------

# IMPORTS:
import pandas as pd

# sklearn libraries
from sklearn.model_selection import train_test_split

# FUNCTIONS:
#---------------------------------------------------------------------
# Function:    process_classifier_data()
# Description: Processes Dataset
#---------------------------------------------------------------------
def process_classifier_data(classifier_csv):
    df = pd.read_csv(classifier_csv, index_col = 0)
    print(df)
    print(df.info())

    # Split Training Data
    df_train, df_test = train_test_split(df, test_size = 0.3)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    # Print Training and Testing Dataset
    print("Training Dataset Count:")
    print(df_train['class_type'].value_counts().sort_index(), "\n")
    print("Testing Dataset Count:")
    print(df_test['class_type'].value_counts().sort_index(), "\n")

    return df_train, df_test

#=====================================================================