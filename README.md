# cds-visual_Assignment_4


***Assignment for visual analytics class at Aarhus University.***

***2021-03-23***


# Classifier benchmarks using Logistic Regression and a Neural Network

## About the script

This assignment is Class Assignment 4. The task was to create two simple command-line tools/Python scripts which can be used to perform a simple classification task on the MNIST digits data.  These classifiers are trained to classify images of digits according to their corresponding labels. After that, the tests are performed and evaluation of the models´ accuracy is coducted:
- One script takes the full MNIST data set, trains a Logistic Regression Classifier, prints the evaluation metrics to the terminal and saves classification report and confusion matrix in a directory
- Another script takes the full MNIST dataset, trains a Neural Network classifier, prints the evaluation metrics to the terminal, and saves classification report in a directory

These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

## Methods

The problem of the task relates to classifying digits. To address this problem, first I trained a simple Logistic Regression classifier on a training set (80% of the data) and tested the performance of the classifier on a test set (20% of the data). For the second script, I trained a Neural Network (NN) using a utility script which was developed in class (it can be found in the utils folder on this github repository). Trained NN had 2 hidden layers out of possible 3 layers. The sizes of the layers were 32 and 16 with a sigmoid activation function. The  NN was trained for 500 epochs.

## Repository contents

| File | Description |
| --- | --- |
| out | Folder containing files produced by the scripts |
| out/logReg_confusion_matrix.png | Confusion matrix of LR classifier |
| out/logReg_report.csv | Classification metrics of the LR classifier |
| out/NN_report.csv | Classification metrics of the NN classifier |
| src | Folder containing the scripts |
| src/Logistic_Regression.py | Logistic Regression classifier script |
| src/Neural_Network.py | Neural Network classifier script |
| utils/ | Folder containing utility scripts for the project  |
| utils/classifier_utils.py | utility script used in LR classifier script |
| utils/neuralnetwork.py | utility script used in NN classifier script |
| README.md | Description of the assignment and the instructions |
| create_classification_venv.bash | bash file for creating a virtual environmment |
| kill_classification_venv.bash | bash file for removing a virtual environment |
| requirements.txt | list of python packages required to run the script |


## Data

The MNIST database contains small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. A training set contains 60,000 examples, and a test set 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

The data was loaded using command ```fetch_openml```. The dataset name is  mnist_784, and version=1.

Link to data: http://yann.lecun.com/exdb/mnist/


## Intructions to run the codes

Both codes were tested on an HP computer with Windows 10 operating system. They were executed on Jupyter worker02.

__Codes parameters__


```Logistic Regression classifier```       

| Parameter | Description |                                              
| --- | --- |                                                                    
| train_size (trs) | The size of the training data as a percentage. Default = 0.8 (80%) |                                       
| test_size (tes) | The size of the testing data as a percentage. Default = 0.2 (20%) | 
| name (n) | Name of the classification report to be saved as .csv file. Default = logReg_report |                                       


```Neural Network classifier```
 
| Parameter | Description |                                              
| --- | --- |                                                                    
| train_size (trs) | The size of the training data as a percentage. Default = 0.8 (80%) |                                       
| test_size (tes) | The size of the testing data as a percentage. Default = 0.2 (20%) | 
| hidden_layer_1 (hl1) | Size of the hidden layer 1. Default = 32 |
| hidden_layer_2 (hl2) | Size of the hidden layer 2. Default = 16 |                               
| hidden_layer_3 (hl3) | Size of the hidden layer 3. Default = 0 |  
| epochs (ep) | Defines how many times the learning algorithm will work through the entire training dataset. Default = 500 |
| name (n) | Name of the classification report to be saved as .csv file. Default = NN_report |
 
Note: In order to define only hidden_layer_1, user must input hidden_layer_2 as 0.



__Steps__

Set-up:
```
#1 Open terminal on worker02 or locally
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-visual_Assignment_4.git 

#4 Navigate to the newly cloned repo
$ cd cds-visual_Assignment_4

#5 Create virtual environment with its dependencies and activate it
$ bash create_classification_venv.sh
$ source ./classification/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the script
$ cd src

#7 Run each code with default parameters
$ python Logistic_Regression.py
$ python Neural_Network.py

#8 Run each code with self-chosen parameters
$ python Logistic_Regression.py -trs 0.9 -tes 0.1 -n lr_cm.csv
$ python Neural_Network.py -trs 0.7 -tes 0.3 -hl1 30 -hl2 15 -hl3 5 -ep 500 -n classification_report

#9 Run the NN script only with hidden_layer_1:
$ python Neural_Network.py -hl1 30 -hl2 0

#10 To remove the newly created virtual environment
$ bash kill_classification_venv

#11 To find out possible optional arguments for both scripts
$ python Logistic_Regression.py --help
$ python Neural_Network.py --help


 ```

I hope it worked!


## Results

Logistic regression classifier achieved a weighted average accuracy of 92% for correctly classifying digits. Digits 3, 5 and 8 were the most challenging to classify. The Neural Network classifier achieved a weighted average accuracy of 96%, which is a slight improvement from LR classifier.
For more information consult classification reports and confusion matrices in the 'out' folder.



