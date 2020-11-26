from random import choice
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def LGBM_param():
    """
    Defining the hyperparameter dictionary for LGBM
    """
    param_dictionary = {
        'num_leaves': int(choice(list(np.linspace(10, 40)))),
        'learning_rate': choice([0.001, 0.002, 0.005, 0.01, 0.05, 0.1]),
        'use_missing': True,
        'max_depth': int(choice(list(np.linspace(1, 10)))),
        'n_estimators': int(choice(list(np.linspace(200, 500, 25)))),
        'objective': 'binary',
    }
    model = LGBMClassifier
    return(param_dictionary, model)

def KNN_param():
    """
    Defining the hyperparameter dictionary for KNN
    """

    param_dictionary = {
        "leaf_size" : int(choice(list(range(1, 50)))),
        "n_neighbors" : int(choice(list(range(1, 30)))),
        "p": int(choice([1, 2])),
    }
    model = KNeighborsClassifier

    return(param_dictionary, model)


def LR_param():
    """
    Defining the hyperparameter dictionary for Logistic Regression
    """
    param_dictionary = {
        'max_iter': int(choice ([50, 100, 200])),
        'C' : choice([0.01, 0.10, 0.1, 10, 25, 50, 100]),
        'random_state': 1,
    }
    model = LogisticRegression
    return(param_dictionary, model)


def MLP_param():

    """
    Defining the hyperparameter dictionary for MLP
    """
    param_dictionary = {
        'hidden_layer_sizes': choice([(50, 50, 50), (50, 100, 50), (100,)]),
        'activation': choice(['tanh', 'relu']),
        'solver': choice(['sgd', 'adam']),
        'alpha': choice([0.005, 0.002, 0.01,0.2, 0.03, 0.05]),
        'learning_rate': choice(['constant','adaptive']),
        "max_iter": 100,
    }
    model = MLPClassifier
    return(param_dictionary, model)





def SVM_param():
    """ 
    Defining the hyperparameter dictionary for SVM
    """
    param_dictionary = {
        'gamma': choice([1e-2, 1e-3, 1e-4, 1e-5]),
        'C': choice([0.01, 0.10, 0.1, 10, 25, 50, 100]),
        'probability': True,
    }

    model = SVC
    return(param_dictionary, model)


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
