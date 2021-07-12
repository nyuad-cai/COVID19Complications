from random import choice
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def LGBM_param():

    param_dictionary_ = {
        'num_leaves': int(choice( list(range(11, 60,10)))),
      
        'use_missing': True,
        'learning_rate': choice(list(np.logspace(-3, 0, 4))),
        'max_depth': int(choice(list(range(10, 60,10)))),
        'num_iterations': int(choice(list(range(100,500,100)))),
        'objective': 'binary',
        
    }
    model = LGBMClassifier
    return(param_dictionary_, model)

def KNN_param():
    param_dictionary_KNN = {
        "leaf_size" : int(choice(list(range(20, 60,5)))),
        "n_neighbors" : int(choice(list(range(5,20,5)))),
        "p": int(choice([2])),
    }
    model = KNeighborsClassifier

    return(param_dictionary_KNN, model)


def LR_param():
    param_dictionary_LR = {
        
        'C': choice(np.logspace(-3, 1, 5)),
        'max_iter' :  choice(list(range(50, 300, 50))),
        "verbose": 0
    }
    model = LogisticRegression
    return(param_dictionary_LR, model)


def SVM_param():
    param_dictionary_SVM = {
        'gamma': choice(list(np.logspace(-4, 1, 6))),
        'C':  choice(list(np.logspace(0, 1, 5))),
        'probability': True,
    }

    model = SVC
    return(param_dictionary_SVM, model)


def choose_baseline(baseline, X):

    if (baseline == "Logistic Regression"):
        param_dictionary, model =  LR_param()
        X = X.fillna(X.median())

    elif (baseline == "LGBM"):
        param_dictionary, model = LGBM_param()

    elif (baseline == "SVM"):
        param_dictionary, model = SVM_param()
        X = X.fillna(X.median())
    elif baseline == "KNN":
        param_dictionary, model = KNN_param()
        X = X.fillna(X.median())

    elif baseline == "MLP":
        param_dictionary, model = MLP_param()
        X = X.fillna(X.median())

    return(param_dictionary, model, X)