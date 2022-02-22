## Imports
import dataprep as dp
import fitpred as fp

if __name__ == '__main__':
    dp.dataPrep(True)

    svm_params = {'C': [1, 10, 100, 1000],
                  'gamma': [0.001, 0.0001],
                  'kernel': ['rbf', 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed']}
    gridSearch(svm_params, svm.SVC())
