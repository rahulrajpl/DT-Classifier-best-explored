"""Trainer program for Decision Tree Classifier.

Program trains a decision tree classifier available from sklearn library.
Training is undertaken using 3 folded RandomizedSearch cross validation of 
random 75% of available dataset. Balance 25% percent is used to test the model.
If the required accuracy is not achieved then program automatically runs again.

Pre-requisites:
--------------
	1. Python 3.6 or above installed. 
		Install it as per instructions in official documentation of python
	2. Python3-tk installed. 
		Install using command 
		>> sudo apt-get install python3-tk
	3. 'virtualenv' installed. 
		Install using command 
		>> sudo apt-get install virtualenv
	4. 'pip' installed. 
		Install using command 
		>> sudo apt-get install -y python3-pip

	5. It is recommended that a virtual environment be created before testing this program
	Same can be created and activated using following command:

		>> virtualenv venv
		>> source venv/bin/activate

	6. All dependencies of this program is mentioned in the requirements.txt file.
	Ensure that they are installed using the command given below:

		>> pip install -r requirements.txt
	
	Note: This program is developed and tested only on ubuntu 18.04. 	
	
Authors:
-------- 
	Sai Charan (pvcharan@iitk.ac.in)
	Rahul Raj (rahulr@iitk.ac.in)
	Khalid Parvez (khalid@iitk.ac.in)


Dept of Computer Science and Engineering
IIT Kanpur (c) 2019

"""


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as  plt
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, f1_score, precision_score
import time, warnings, sys, glob, joblib, concurrent.futures
warnings.filterwarnings('ignore')

def generate_data():
    """Generate random dataset from the data file.

    Retrieves random rows from dataset of the file 'HW_TESLA.xlt'
    after clearning and splitting them into customary datasets 
    such as X_train, y_train, X_test, y_test
  
    Args:
    -----
        None 
          
    Returns: 
    --------
        A tuple with following objects in same order
        
        X_train :   Feature matrix of given samples for training
        y_train :   Labels of corresponding samples for training 
        X_test  :   Feature matrix of given samples for test
        y_test  :   Labels of corresponding samples for test
    
    Example usage
    -------------
	>> X_train, y_train, X_test, y_test = generate_data()
        
    """
    df = pd.read_excel('HW_TESLA.xlt')
    le=LabelEncoder().fit_transform(df['STATIC'])
    df_train, df_test = train_test_split(df, test_size=0.25) # 75 % training data
    # Separating feature matrix and corresponding labels
    X_train = df_train.drop('STATIC',axis=1)
    X_test = df_test.drop('STATIC',axis=1)
    y_train = df_train['STATIC']
    y_test = df_test['STATIC']
    return X_train, y_train, X_test, y_test

def generate_report(conf_mat, y_test, y_pred):
    """ 
    This function will generate report with False Positive, False Negative, accuracy, recall, precision and f1_score
        
    Args: 
    -----
        conf_mat    : Confusion Matrix
        y_test      : labels of test data set
        y_pred      : labels predicted on test data set 
        
    Returns: 
    --------
        Returns following data a tuple in the following order

        FP (int)        : Number of Flase Positives
        FN (int)        : Number of Flase Negative
        accuracy (float): Prediction accuracy calculated using sklearn's accuracy_score() function
        recall (float)  : Recall calculated using sklearn's recall() function
        precision(float): Precision calculated using sklearn's precision() function
        f1score (float) : F1Score calculated using sklearn's f1score() function
    
    Example usage
    -------------
        >> FP, FN, accuracy, recall, precision, f1score = generate_report(conf_mat, y_test, y_pred)

    """
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    return FP, FN, accuracy, recall, precision, f1score


def train_dt(X_train, y_train, X_test, y_test): 
    """Training DecisionTree Classifier of sklearn

    This function will train the DT Classifier imported from sklearn library
    by utilizing the randomizedsearch cross validation technique. Parameters 
    set for the randomized search is as follows:

               'criterion': ['gini','entropy']
                'max_depth': [2,3,4,5,6,7,8,9]
                'min_samples_split':[2,3,4,5]
    
    A Note of Scoring criteria:
    It may be noted that by default the randomized search will optimize and 
    tune the hyperparameters for best prediction accuracy. But this has been 
    modified to 'prediction accuracy' as well as 'recall'. In this particular 
    example, we have to ensure minimum false negetive figures in the overall 
    performance of the classifier. Thus, scoring criteria for parameter tuning 
    is selected as follows:

    scoring = ['accuracy', 'recall'], refit='accuracy'

    Enabling maximum cpu_utilization:
    n_jobs parameter of randomized search is set to -1, which indicates that 
    all available CPU cores will be engaged to maximum capacity

    Args:
    -----
        X_train :   Feature matrix of given samples for training
        y_train :   Labels of corresponding samples for training 
        X_test  :   Feature matrix of given samples for test
        y_test  :   Labels of corresponding samples for test 

    Returns:
    --------
        outcome :   tuple with output of generate_report(). See __doc__ string of
                    generate_report() for more details. Additionally, 'outcome' variable
                    carried the cross_validation score calculated using 'cross_val_score'
                    function imported from sklearn as well as the time taken for the 
                    execution of the training in seconds.

                    outcome =
                    (cross_val_score, FP, FN, accuracy, recall, precision, f1score, time_elapsed)

        clf_rs  :   The trained Decision Tree classifier object

    Example usage
    -------------
        >> outcome, clf = train_dt(X_train, y_train, X_test, y_test)

    """
    start = time.perf_counter()
    scoring = ['accuracy', 'recall']
    p_grid = {'criterion': ['gini','entropy'],
               'max_depth': [2,3,4,5,6,7,8,9],
               'min_samples_split':[2,3,4,5],
               }

    clf = DecisionTreeClassifier()
    clf_rs = RandomizedSearchCV(estimator=clf, param_distributions=p_grid,
                                n_iter=500, cv = 3, n_jobs=-1, scoring=scoring , refit='accuracy')
    clf_rs.fit(X_train, y_train)
    scores = cross_val_score(clf_rs, X_train, y_train, cv=3)
    val_score = (scores.mean(),  scores.std() * 2)

    y_pred = clf_rs.predict(X_test)
    conf_mat=confusion_matrix(y_test,y_pred)
    outcome = list(generate_report(conf_mat, y_test, y_pred))
    outcome.insert(0, val_score[0])
    finish = time.perf_counter()
    outcome.append(str(round(finish-start ,2))+' Secs')
    return outcome, clf_rs

def start_training():
    """Initialize training with specified target model performance

    This function intializes the train_dt() function with a specified criteria
    of prediction accuracy and False negative figure. See doc string of train_dt()
    for more details on it. Variables indicating them with
    default values are as follows:
        
        expected_accuracy = 0.997
        expected_FN = 2

    (This values should be edited to obtain model performance as per requirement.)
    CAUTION: Higher accuracy figure or lower FN number will take longer time to train.

    Args:
    -----
        None

    Returns:
    --------
        measurements    : List of all the outcomes returned by train_dt() function 

    Example Usage:
    --------------
        >> m = start_training()
    """
    expected_accuracy = 0.997
    expected_FN = 2
    # expected_FP = 1
   
    X_train, y_train, X_test, y_test = generate_data()        

    measurements = []
    outcome, clf = train_dt(X_train, y_train, X_test, y_test)
    measurements.append(outcome)
    # outcome[2] -> False Negative and  outcome[3] -> Prediction Accuracy
    if outcome[3] > expected_accuracy and outcome[2] <= expected_FN:
        filename = 'train_dt' + '_model.model'
        joblib.dump(clf, filename)  # Save the model for future use
    return measurements[0]


if __name__ == "__main__":    

    f1 = 'train_dt_model.model' # File name for persistant storage
    cols = ['Val_Accuracy', 'FP', 'FN', 'Accuracy', 'Recall', 'Precision', 'F1Score', 'Time_elapsed']
    message = 'Training in progress with a random dataset for each iteration..please wait'
    outcome = []
    

    if f1 in glob.glob("*.model"):
        print('Exiting Program \nPre trained model already available. Check the file:', f1)
        exit(1)
    t1 = time.perf_counter()
    print(message)
    sys.stdout.write('Training Progress:[')
    while True:
        sys.stdout.write('#')
        m = start_training()
        outcome.append(m)
        if f1 in glob.glob("*.model"):
            t2 = time.perf_counter()
            t = ']Total time elapsed: ' + str(round((t2-t1), 2)) + ' Seconds'
            sys.stdout.write(t)
            sys.stdout.write('\n')
            print('\n\n @@..Trained a model with expected accuracy..@@\nHistory of experiments as follows \n')
            break
    
    
    df_outcome = pd.DataFrame(outcome, columns=cols)
    print(df_outcome)
    print('Experiment Completed')
