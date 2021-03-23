#!/usr/bin/env python

'''
Assignment 4: Classification benchmarks

This script takes a subset (8x8) of MNIST dataset, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. Optionally, it can save a regression classifier report as csv.

Optional argument:
    "-n", "--name":   name of the out file


Example:
    saving .csv file:
        $ python Logistic_Regression.py -n lm_cm.csv
    
    not saving .csv file
        $ python Logistic_Regression.py


'''

## Importing libraries

# System tools
import os
import sys
sys.path.append(os.path.join(".."))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import pandas for working with dataframes
import pandas as pd

# Import sklearn metrics
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Command-line interface
import argparse



# Defining the main script

def main():
    # instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser()
    # adding optional arguments
    parser.add_argument("-n", "--name", required=False, help="name of the out file")
    # parsing the arguments
    args = vars(parser.parse_args())

## Loading data and preprocessing
    
    # Load sample data
    digits = datasets.load_digits()   
    # Convert to floats
    data = digits.data.astype("float")
    # MinMax regularization
    data = (data - data.min())/(data.max() - data.min())
    # Creating train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                  digits.target, 
                                                  test_size=0.2)
    # Scaling the features
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    
    
## Training a logistic regression model
    
    # Defining a model
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    

## Evaluating the model: accuracy measures and classifier report
    
    # Predicted labels
    y_pred = clf.predict(X_test_scaled)
    # Calculating  accuracy of the model based on comparing the predicted labels with the actual labels
    accuracy = accuracy_score(y_test, y_pred)
    # Logistic Regression classifier report 
    cm = metrics.classification_report(y_test, y_pred)
    
    
    
## If user inputs optional argument, save .csv file:

    if args["name"]:
        
        #Create ouput folder for saving the classifier report if it doesn´t exist already
        if not os.path.exists("out"):
            os.makedirs("out")
        
        # Turning classifier report into a dataframe
        report_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True)).transpose()    
        # defining full filepath to save csv file 
        outfile = os.path.join("out", args["name"])
        # saving a dataframe as .csv
        report_df.to_csv(outfile)  
        
        
    
    # Displaying the final messages and results       
    print("SUCCESS! Logistic regression model has worked and was evaluated. Full evaluation of the model is displayed below")
    print(f"Accuracy of the current logistic regression model is = {accuracy}")
    print(cm)
    
    
     
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    

