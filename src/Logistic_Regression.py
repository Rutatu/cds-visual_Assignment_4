#!/usr/bin/env python

''' ---------------- About the script ----------------

Assignment 4: Classification benchmarks: Logistic Regression

This script takes a MNIST digits dataset, trains a Logistic Regression Classifier, prints the evaluation metrics to the terminal and saves classification report and a confusion matrix to a directory. Optionally, user can define both training data and test data sizes, name of a classification report file. There are no positional arguments, only default values of optional arguments.

Optional argument:
    -trs, --train_size:       The size of the training data as a percentage, where the default = 0.8 (80%)
    -tes, --test_size:        The size of the test data as a percentage, where the default = 0.2 (20%)
    -n,   --name:             Name of the classification report to be saved as .csv file, where the default = logReg_report


Example:
    customized optional arguments:
        $ python Logistic_Regression.py -trs 0.9 -tes 0.1 -n lr_cm.csv
    
    default arguments:
        $ python Logistic_Regression.py


'''



"""---------------- Importing libraries ----------------
"""

# System tools
import os
import sys
sys.path.append(os.path.join(".."))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import pandas for working with dataframes
import pandas as pd

# visualization tool
import matplotlib.pyplot as plt

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Command-line interface
import argparse




"""---------------- Main script ----------------
"""

def main():
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description = "[INFO] Classify MNIST data and print out performance accuracy report")
    
    # Adding optional arguments                                                  
    parser.add_argument("-trs", "--train_size", required = False, default = 0.8, type = float, help = "The size of the training data as a percentage. Default = 0.8 (80%)")
    parser.add_argument("-tes", "--test_size", required = False, default = 0.2, type = float, help = "The size of the testing data as a percentage. Default = 0.2 (20%)")
    parser.add_argument("-n", "--name", required=False, default = "logReg_report", help="Name of the classification report to be saved as .csv file. Default = logReg_report")
                                     
    # Parsing the arguments
    args = vars(parser.parse_args())
                                     
                                     


    """------ Loading full data and preprocessing ------
    """
        
    print("[INFO] Loading and preprocessing data...")
    
    # Fetching data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    
    # Converting to numpy arrays
    X = np.array(X) # Data
    y = np.array(y) # Classes (0-9)
    
    
    # Creating training data and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size = args["train_size"],
                                                        test_size= args["test_size"])
    
    # Scaling the features from 0-255 to between 0 and 1
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    
    
    """------ Loading sample data and preprocessing ------
    """
    
    # Load sample data
    ##digits = datasets.load_digits()   
    # Convert to floats
    ##data = digits.data.astype("float")
    # MinMax regularization
    ##data = (data - data.min())/(data.max() - data.min())
    # Creating train and test datasets
    ##X_train, X_test, y_train, y_test = train_test_split(data, 
                                                  ##digits.target,
                                                  ##train_size = args["train_size"],
                                                  ##test_size= args["test_size"])
    
    
    
    
    """------ Training the model ------
    """
    
    # Defining a model
    print("[INFO] training Logistic Regression model...")
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train) #Fitting classifier to test data
    

    
    """------ Evaluating the model: accuracy measures and classifier report ------
    """

    print("[INFO] evaluating Logistic Regression model...")
    # Predicted labels
    y_pred = clf.predict(X_test_scaled)
    # Calculating  accuracy of the model based on comparing the predicted labels with the actual labels
    accuracy = accuracy_score(y_test, y_pred)
    # Logistic Regression classifier report 
    cm = metrics.classification_report(y_test, y_pred)
    
    
    
    """------ Saving classification report as .csv file ------
    """
    
 
    # Create ouput folder for saving the classification report if it doesn´t exist already
    if not os.path.exists("../out"):
        os.makedirs("../out")
        
    # Turning classification report into a dataframe
    report_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True)).transpose()    
    # Defining full filepath to save csv file 
    outfile = os.path.join("..", "out", args["name"])
    # Saving a dataframe as .csv
    report_df.to_csv(outfile)  
    # Printing that .csv file has been saved
    print(f"\n[INFO] classification report is saved in directory {outfile}")
        
    """------ Saving confusion matrix as .png file ------
    """
    
    # Defining full filepath to save .png file
    path_png = os.path.join("..", "out", "logReg_confusion_matrix.png")
    # Creating confusion matrix
    clf_util.plot_cm(y_test, y_pred, normalized=True)
    # Saving as .png file
    plt.savefig(path_png)
    # Printing that .png file has been saved
    print(f"\n[INFO] confusion matrix is saved in directory {path_png}")
    
    
    """------ Final messages and printing results ------
    """
                                     
    print("\nSUCCESS! Logistic regression model has been trained and evaluated. Full evaluation of the model is displayed below")
    print(f"\nWeighted average accuracy of the current logistic regression model is = {accuracy}")
    print(cm) # full classification report
    
    
                                     

        
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    

