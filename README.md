# cds-visual_Assignment_4


***Assignment for visual analytics class at Aarhus University.***

***2021-03-23***


# Classification benchmarks






# Instructions to run the code


Preparation:

 - Open terminal on worker02
 - Navigate to the environment where you want to clone this repository, e.g. type: cd cds-visual
 - Clone the repository, type: git clone https://github.com/Rutatu/cds-visual_Assignment_4.git
 - Navigate to the newly cloned repo, type: cd cds-visual_Assignment_4
 - Create virtual environment with its dependencies, type: bash create_classification_venv.sh
 - Activate the environment, type: source ./classification/bin/activate
 - Continue with running the scripts, see intructions below

Running the scripts:

 - Navigate to the directory of the script, type: cd src
 - Run Logistic regression model script, type:  python Logistic_Regression.py
 - Run Neural Network script, type: python Neural_Network.py
 - To find out possible optional arguments for both scripts, type: python Logistic_Regression.py --help   or   python Neural_Network.py --help

 - To remove newly created virtual environment from worker02, type: bash kill_classification_venv




