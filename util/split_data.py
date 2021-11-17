#!/usr/bin/env python3

import sys
import os.path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FILE_FORMAT="{0}_{2}{1}.csv"

def get_filepath(output_directory, data_type, size, note=""):
    return os.path.join(output_directory, FILE_FORMAT.format(data_type, int(size*100), note))

def split_data(input_filepath, output_directory, test_size=0.2):
    data = pd.read_csv(input_filepath)
    y = data.pop('Attrition_Flag')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    training_data = pd.concat([X_train, y_train], axis=1)

    training_data.to_csv(get_filepath(output_directory, "training", 1-test_size, note="stratified"), index=False)

    X_test.to_csv(get_filepath(output_directory, "testing", test_size, note="stratified"), index=False)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect usage. Please include the path to the input file and the path to the output directory.\nExample: python3 util/split_data.py data/full.csv data")
    else:
        split_data(sys.argv[1], sys.argv[2])