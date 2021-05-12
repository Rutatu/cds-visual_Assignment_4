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
| out/ | Classification metrics of the model |
| out/ | Model´s performance graph |
| out/ | Depiction of CNN model´s architecture used |
| src | Folder containing the script |
| src/ | The script |
| README.md | Description of the assignment and the instructions |
|  | bash file for creating a virtual environmment  |
|  | bash file for killing a virtual environment |
| requirements.txt | list of python packages required to run the script |


## Data

For this project I used a MNIST database containing handwritten digits with a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

The data was loaded using command ```fetch_openml```. The dataset name is  "mnist_784", and version=1.

Link to data: http://yann.lecun.com/exdb/mnist/


## Intructions to run the codes

Both codes were tested on an HP computer with Windows 10 operating system. They were executed on Jupyter worker02.

__Codes parameters__

Logistic Regression classifier:                                                    Neural Network classifier:

| Parameter | Description |                                                        | Parameter | Description |
| --- | --- |                                                                      | --- | --- |
| train_data | Directory of training data |                                        | train_data | Directory of training data |
| val_data | Directory of validation data |                                        | val_data | Directory of validation data |
| learning_rate | Learning rate. Default = 0.01 |                                  | learning_rate | Learning rate. Default = 0.01 |
| optimizer | Optimizer. Default = SGD |                                           | optimizer | Optimizer. Default = SGD |
| epochs | Number of epochs. Default = 50 |                                        | epochs | Number of epochs. Default = 50 |





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
$ python 

#8 Run each code with self-chosen parameters
$ python 

#9 To remove the newly created virtual environment
$ bash kill_classification_venv

#10 To find out possible optional arguments for both scripts
$ python Logistic_Regression.py --help
$ python Neural_Network.py --help


 ```

I hope it worked!


## Results

Logistic regression classifier achieved a weighted average accuracy of 92% for correctly classifying digits. Digits 3, 5 and 8 were the most challenging to classify. The Neural Network classifier achieved a weighted average accuracy of 96%, which is a slight improvement from LR classifier.
For more information consult classification reports and confusion matrices in the 'out' folder.



