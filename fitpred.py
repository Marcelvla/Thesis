## Imports
import dataprep as dp
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import scipy as sp
import time

## Functions
def simpleFitPredict(clfs):
    ''' Create a simple fit to an SVM and predict papers for the testset
    '''
    X, y, X_test, y_test = dp.dataPrep()
    clf_list = []
    pred_list = []
    for clf in clfs:
        print(f"Fitting {str(clf)}", end="\r")
        start = time.time()

        if str(clf) == "MLPClassifier()" or str(clf) == "GaussianNB()":
            clf_list.append(clf.fit(X.toarray(), y))
            print(f"Fitted {str(clf)} in {time.time() - start} seconds", end="\r")
            pred_list.append(clf.predict(X_test.toarray()))
        else:
            clf_list.append(clf.fit(X, y))
            print(f"Fitted {str(clf)} in {time.time() - start} seconds", end="\r")
            pred_list.append(clf.predict(X_test))
        end = time.time()
        print(f"Fit&predict {str(clf)} in {end-start} seconds")

    return clf_list, pred_list, y_test

def printMetrics(clf_str, y_pred, y_test):
    ''' Show scores for the predicted values of a classifier
    '''
    # confusion matrix on the test data.
    print(f'\nConfusion matrix of {clf_str} on the test data:')
    pred_col_index = list(set(y_pred))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred)), columns = pred_col_index, index = pred_col_index)
    pred_ct = Counter([y_pred[i]==y_test[i] for i in range(len(y_test))])
    acc = pred_ct[True] / len(y_test) * 100
    print(f"With an accuracy of {acc}%")

    return acc

def gridSearch(parameters, clf):
    ''' Implement a gridsearch and return the score for
        supposed optimal hyper-parameters
    '''
    print("Getting test and train data...")
    X_train, y_train, X_test, y_test = dp.dataPrep(True)
    print("Done!")
    gridclf = GridSearchCV(clf, parameters)
    print("Fitting training data...")
    gridclf.fit(X_train, y_train)
    print("Fitted!")
    print("Predicting test data...")
    y_pred = gridclf.predict(X_test)
    print("Done!")
    print(gridclf.best_params_)
    printMetrics('SVM classifier', y_pred, y_test)
