#!/usr/bin/env python

'''

Assignment 4: Classification benchmarks

This script takes a subset (8x8) of MNIST dataset, trains a Neural Network model, and prints the evaluation metrics to the terminal. Optionally, you can choose how many hidden layers you want to include.

Optional argument:
    "-hl1", "--hidden_layer_1":   size of the hidden layer
    "-hl2", "--hidden_layer_2":   size of the hidden layer
    "-hl3", "--hidden_layer_3":   size of the hidden layer



Example:
    only hidden_layer_1:
        $ python Neural_Network.py -hl1 30
    
    default hidden layers (2 of them)
        $ python Neural_Network.py



'''

## Importing libraries

# System tools
import sys 
import os
sys.path.append(os.path.join(".."))

# Import pandas for working with dataframes
import pandas as pd

# Neural networks with numpy
from utils.neuralnetwork import NeuralNetwork

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Command-line interface
import argparse



# Defining the main script
def main():
    # instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser()
    # adding optional arguments
    parser.add_argument("-hl1", "--hidden_layer_1", help="size of the hidden layer", type = int)
    parser.add_argument("-hl2", "--hidden_layer_2", help="size of the hidden layer", type = int)
    parser.add_argument("-hl3", "--hidden_layer_3", help="size of the hidden layer", type = int) 
    
   
      
    # When including 'name' argument and in case user inputs only this argument, the script starts runing from the line 'if args["name"]:', which causes problems, cause I need to run neural network twice, and I did not come up with elegant solution to do so. It is even more problematic when user inputs other arguments along with 'name' argument.
    #parser.add_argument("-n", "--name", help="name of the out file")

    # parsing the arguments
    args = parser.parse_args()
    
    

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
    # convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    

    
    
### Optional arguments 
           

## If user inputs 1 hidden layer:
    
    if args.hidden_layer_1:
        
        ## Training a neural network

        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=1000)
   
    
        ## Evaluating the network

        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))
    
      
## If user inputs 2 hidden layers: 
        
    if args.hidden_layer_1 and args.hidden_layer_2:
        
        ## Training a neural network

        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, args.hidden_layer_2, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=1000)
   
    
        ## Evaluating the network

        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))
        

        
## If user inputs 3 hidden layers:

    if args.hidden_layer_1 and args.hidden_layer_2 and args.hidden_layer_3:
        
        ## Training a neural network

        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], args.hidden_layer_1, args.hidden_layer_2, args.hidden_layer_3, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=1000)
   
    
        ## Evaluating the network

        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))
        

## If user doesn´t input any hidden layers:        
        
    else:
    
        ## Training a neural network

        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=1000)
   

        ## Evaluating the network

        print(["[INFO] evaluating network..."])
        predictions = nn.predict(X_test)
        predictions = predictions.argmax(axis=1)
        print(classification_report(y_test.argmax(axis=1), predictions))

    
    print("Have a nice day!")
    
    
    
    ### DIDN`T WORK ###
    
    ## If user inputs optional argument 'name', save .csv file:

    #if args.name:
        
        
        #Create ouput folder for saving the classifier report if it doesn´t exist already
        #if not os.path.exists("out"):
            #os.makedirs("out")
        
        
        # Turning classifier report into a dataframe
        #report_df = pd.DataFrame(classification_report(y_test.argmax(axis=1), predictions, output_dict = True)).transpose()    
        # defining full filepath to save csv file 
        #outfile = os.path.join("out", args["name"])
        # saving a dataframe as .csv
        #report_df.to_csv(outfile) 
        


        
        
         
            
# Define behaviour when called from command line

if __name__=="__main__":
    main()
 
    
    
    
