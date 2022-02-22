## Imports
import dataprep as dp
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

## Functions
def simpleFitPredict(clf):
    ''' Create a simple fit to an SVM and predict papers for the testset
    '''
    X, y, X_test, y_test = dp.dataPrep()
    clf.fit(X,y)

    return clf.predict(X_test)

def printMetrics(clf_str, y_pred, y_test):
    ''' Show scores for the predicted values of a classifier
    '''
    # confusion matrix on the test data.
    print(f'\nConfusion matrix of {clf_str} on the test data:')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

def gridSearch(parameters, clf):
    ''' Implement a gridsearch and return the score for
        supposed optimal hyper-parameters
    '''
    print("Getting test and train data...")
    X_train, y_train, X_test, y_test = dp.dataPrep()
    print("Done!")
    gridclf = GridSearchCV(clf, parameters)
    print("Fitting training data...")
    gridclf.fit(X_train, y_train)
    print("Fitted!")
    print("Predicting test data...")
    y_pred = gridclf.predict(X_test)
    print("Done!")
    print(gridclf.get_params())
