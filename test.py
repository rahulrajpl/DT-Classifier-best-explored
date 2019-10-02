import joblib, glob, sys
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from train import generate_report

usage = '>>> test.py n-samples n-experiments \n Example \n >>> test.py 200 10'
models_list = ['linear_svm', 'tree', 'logistic']
if(len(sys.argv) < 3):
    print('ERROR!\n Specify the number of samples and experiments to be done in following format\n', usage)
    exit(1)

model_file = glob.glob("train_dt_model.model")
# model_file = glob.glob("Max_Acc_1 copy.sav")


try:
    pre_trained_model = joblib.load(model_file[0])
    df = pd.read_excel('HW_TESLA.xlt')
    df_results = []
    n_samples = int(sys.argv[1])
    n_experiments = int(sys.argv[2])

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
    print("Training not done!. First run the file train.py to generate model for testing")
    exit(1)

except IndexError:
    print('ERROR!')
    print("Index Error in calculating confusion matrix. Kindly change sample size")
    exit(1)

except:
    print('ERROR!')
    print("-------------")
    exit(1)




