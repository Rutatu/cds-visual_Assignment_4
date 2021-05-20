#!/usr/bin/env python

""" ---------------- About the script ----------------

Assignment 4: Classification benchmarks: Neural Network

This script takes a MNIST dataset, trains a Neural Network model, and prints the evaluation metrics to the terminal. Optionally, you can choose how many hidden layers you want to include and whether to save a classification report as .csv file. There are no positional arguments, only default values of optional arguments.

Optional arguments:
    -trs, --train_size:       The size of the training data as a percentage, where the default = 0.8 (80%)
    -tes, --test_size:        The size of the test data as a percentage, where the default = 0.2 (20%)
    -hl1, --hidden_layer_1:   Size of the hidden layer 1, where the default = 32
    -hl2, --hidden_layer_2:   Size of the hidden layer 2, where the default = 16
    -hl3, --hidden_layer_3:   Size of the hidden layer 3, where the default = 0
    -ep,  --epochs:           The number times that the learning algorithm will work through the entire training dataset, where the default = 1000
    -n,   --name:             Name of the classification report to be saved as .csv file, where the default = 

    Note: In order to define only --hidden_layer_1, user must input --hidden_layer_2 as 0.


Example:

    run the script with default arguments:
        $ python Neural_Network.py
        
    run the script only with hidden_layer_1:
        $ python Neural_Network.py -hl1 30 -hl2 0
   
    run the script with optional arguments:
        $ python Neural_Network.py -trs 0.7 -tes 0.3 -hl1 30 -hl2 15 -hl3 5 -ep 500 -n classification_report
        
"""



"""---------------- Importing libraries ----------------
"""

# System tools
import sys 
import os
sys.path.append(os.path.join(".."))

# Import pandas for working with dataframes
import pandas as pd

# Neural networks with numpy
import numpy as np
from utils.neuralnetwork import NeuralNetwork

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import fetch_openml

# Command-line interface
import argparse




"""---------------- Main script ----------------
"""
    
def main():
    
    """------ Argparse parameters ------
    """
    # instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser()
    # adding optional arguments
    parser.add_argument("-trs", "--train_size", default = 0.8, type = float, help = "The size of the training data as a percentage. Default = 0.8 (80%)")
    parser.add_argument("-tes", "--test_size",  default = 0.2, type = float, help = "The size of the training data as a percentage. Default = 0.2 (20%)")
    parser.add_argument("-hl1", "--hidden_layer_1", default = 32, type = int, help="Size of the hidden layer 1. Default = 32")
    parser.add_argument("-hl2", "--hidden_layer_2", default = 16, type = int, help="Size of the hidden layer 2. Default = 16")
    parser.add_argument("-hl3", "--hidden_layer_3", default = 0, type = int, help="Size of the hidden layer 3. Default = 0") 
    parser.add_argument("-ep", "--epochs", default = 500, type = int, help = "Defines how many times the learning algorithm will work through the entire training dataset. Default = 500")
    parser.add_argument("-n", "--name", default = "NN_report", help="Name of the classification report to be saved as .csv file")
     

    # parsing the arguments
    args = parser.parse_args()
    
    

    
    """------ Loading full data and preprocessing ------
    """
    
    print("[INFO] Loading and preprocessing data...")
    
    # Fetching data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    
    # Converting to numpy arrays
    X = np.array(X) #data
    y = np.array(y) #labels
    
    # MinMax regularization (rescaling from 0-255 to 0-1)
    X = (X - X.min())/(X.max() - X.min())
    
    # Creating training data and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        train_size = args.train_size,
                                                        test_size = args.test_size)
    # Converting labels from integers to vectors (binary)
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    
    
    """------ Loading sample data and preprocessing ------
    """
    
    # Load sample data
    ###digits = datasets.load_digits()   
    # Convert to floats
    ###data = digits.data.astype("float")
    # MinMax regularization (rescaling from 0-255 to 0-1)
    ###data = (data - data.min())/(data.max() - data.min())
    # Creating train and test datasets
    ###X_train, X_test, y_train, y_test = train_test_split(data, 
                                                  #digits.target, 
                                                  #train_size = args.train_size,
                                                  #test_size = args.test_size)
    # Converting labels from integers to vectors
    ###y_train = LabelBinarizer().fit_transform(y_train)
    ###y_test = LabelBinarizer().fit_transform(y_test)
    

    
    """------ Training the network (behavior with optional hidden layers) ------
    """
           

    # If user inputs 1 hidden layer:
    if args.hidden_layer_1 > 0 and args.hidden_layer_2 == 0 and args.hidden_layer_3 == 0:
        
        # Training a neural network
        print("[INFO] training Neural Network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs = args.epochs)
   
      
    # If user inputs 2 hidden layers: 
    elif args.hidden_layer_1 > 0 and args.hidden_layer_2 > 0 and args.hidden_layer_3 == 0:
        
        ## Training a neural network
        print("[INFO] training Neural Network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, args.hidden_layer_2, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs = args.epochs)
   
         
    # If user inputs 3 hidden layers:
    elif args.hidden_layer_1 > 0 and args.hidden_layer_2 > 0 and args.hidden_layer_3 > 0:
        
        ## Training a neural network
        print("[INFO] training Neural Network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, args.hidden_layer_2, args.hidden_layer_3, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs = args.epochs)
   
    

    """------ Evaluating the network ------
    """
      
    # Evaluating the network
    print(["[INFO] evaluating Neural Network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    
        
    
    """------ Saving classification report as .csv file (optional) ------
    """
    # If user inputs optional argument 'name', save as .csv file:
    if args.name:
        
        #Create ouput folder for saving the classification report if it doesn´t exist already
        if not os.path.exists("../out"):
            os.makedirs("../out")
        
        # Turning classification report into a dataframe
        report_df = pd.DataFrame(classification_report(y_test.argmax(axis=1), predictions, output_dict = True)).transpose()    
        # Defining full filepath to save csv file 
        outfile = os.path.join("..", "out", args.name)
        # Saving a dataframe as .csv
        report_df.to_csv(outfile) 
        # Printing that .csv file has been saved
        print(f"\n[INFO] classification report is saved in directory {outfile}")
        
    
    
    """------ Final messages ------
    """

    # Printing a message to the user
    print("The script was executed successfully. Have a nice day!")
    
    
         
            
# Define behaviour when called from command line

if __name__=="__main__":
    main()
 
    
    
    
