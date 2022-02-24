## Imports
import dataprep as dp
import fitpred as fp
from sklearn import svm, neural_network as nn, tree, naive_bayes as nb
import numpy as np

# Write tests here
def gridSVMtest():
    # Do gridsearch for SVM with parameters
    svm_params = {'C': list(range(50, 250, 50)),
                  'gamma': list(np.arange(0.0005, 0.0025, 0.0005)),
                  'kernel': ['sigmoid']}
    fp.gridSearch(svm_params, svm.SVC())
    # BEST PARAMS: {'C': 50, 'gamma': 0.002, 'kernel': 'sigmoid'}

def simpleFPtest():
    # List of classifiers to use
    # clfs = [nb.GaussianNB()]
    # clfs = [svm.SVC(),
    #         tree.DecisionTreeClassifier(),
    #         nb.GaussianNB(),
    #         nn.MLPClassifier()] #TAKES LONG TO FIT
    clfs = [tree.DecisionTreeClassifier(),
            nb.GaussianNB()]
    clfs_fitted, pred, y_test = fp.simpleFitPredict(clfs)
    scores = [[fp.printMetrics(clfs[i], pred[i], y_test) for i in range(len(clfs))]]

    return pred, y_test, scores

if __name__ == '__main__':
    ## Run tests here
    pred, y_test, scores = simpleFPtest()
