"""Program to test the pretrained model generated with train.py

This program is to be executed after running train.py which will
generate the model and save it for persistent use.

Authors:
-------- 
	Sai Charan (pvcharan@iitk.ac.in)
	Rahul Raj (rahulr@iitk.ac.in)
	Khalid Parvez (khalid@iitk.ac.in)


Dept of Computer Science and Engineering
IIT Kanpur (c) 2019
"""

import joblib, glob, sys
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from train import generate_report

usage = '>>> python3 test.py <nsamples> <niter> \n Example \n >>> python3 test.py 200 10 \n To test with \
best model trained use\n >>> python3 test.py 2000 10 best.model'
models_list = ['linear_svm', 'tree', 'logistic']

if (len(sys.argv) < 3):
    print('ERROR!\n Specify the number of samples and experiments to be done in following format\n', usage)
    exit(1)

elif (len(sys.argv) == 3):
    n_samples = int(sys.argv[1])
    n_experiments = int(sys.argv[2])
    fname = "train_dt_model.model"

elif (len(sys.argv) == 4):
    print("Testing with the best pretrained model!")
    n_samples = int(sys.argv[1])
    n_experiments = int(sys.argv[2])
    fname = sys.argv[3]

model_file = fname

try:
    pre_trained_model = joblib.load(model_file)
    df = pd.read_excel('HW_TESLA.xlt')
    df_results = []
    

    cols = ['FP', 'FN', 'Accuracy', 'Recall', 'Precision', 'F1Score']

    for _ in range(n_experiments):
        df_test = df.sample(n=n_samples)
        X_test = df_test.drop('STATIC', axis=1)
        y_test = df_test['STATIC']
        y_pred = pre_trained_model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        df_results.append(list(generate_report(conf_mat, y_test, y_pred)))

    df_prediction_status = pd.DataFrame(df_results, columns=cols)
    print("Testing Completed for ", n_samples,
        " random samples from data file. \n Result is as follows")
    print(df_prediction_status)

except FileNotFoundError:
    print('ERROR!')
    print("Training not done!. First run the file train.py to generate model for testing\n\
        To test with the best model trained, use\n >>> python3 test.py <nsamples> <niter> best.model")
    exit(1)

except IndexError:
    print('ERROR!')
    print("Index Error in calculating confusion matrix. Kindly change sample size")
    exit(1)

except Exception as e: 
    print(e)
    exit(1)




