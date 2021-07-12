#loading useful libraries
import pandas as pd
import numpy as np
from scipy import stats
import shap
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_results(framework_test, complications, models_all, train_columns):
    """This function applies the system on all the complications,
    returns the predictions and the performance on the test sets 
    """

    upper_bounds_auc = []
    upper_bounds_AUPRC = []
    lower_bounds_auc = []
    lower_bounds_AUPRC = []
    true_aucs = []
    true_auprc = []
    predicted_ouctomes = []
    true_ouctomes = []

    for i in complications:
        outcome = (f"{i}_after24")
        print( f"predicting {i}")
        test = framework_test[i]

        model_chosen = models_all[i]
        
        if type(model_chosen[0]) == LogisticRegression:
            test = test.fillna(test.median())
        elif type(model_chosen[0].base_estimator) != LGBMClassifier:
            print(type(model_chosen[0].base_estimator))

            test = test.fillna(test.median())

        auc_true, AUPRC_true, prediction, upperAUPRC, lowerAUPRC, upperAUC, lowerAUC = apply_system(outcome, test, model_chosen, train_columns)
        print("Calcualing 95% Confidence Intervals")
        print("-------------------------------------------")

        upper_bounds_auc.append(upperAUC)
        upper_bounds_AUPRC.append(upperAUPRC)


        lower_bounds_auc.append(lowerAUC)
        lower_bounds_AUPRC.append(lowerAUPRC)


        true_aucs.append(auc_true)
        true_auprc.append(AUPRC_true)

        predicted_ouctomes.append(prediction)
        true_ouctomes.append(test[outcome])

    return(true_ouctomes, predicted_ouctomes)

def predict_with_bootstraps(test, outcome, models, train_columns):

    """ 
    The function below calculates the avarage predictions of the 6 models per complication 
    then gets boostraps with 1000 iterations to calculate the 95% confidence Intervals. 
    """
    n_iterations = 1000
    Test_AUC=[]
    Test_AUPRC=[]

    # prediction_list = []
    predictions_df = pd.DataFrame()

    count = 0
    for i in models:
        predictions_df[f"prediction{str(count)}"] =i.predict_proba(test[train_columns])[:, 1]
        count = count +1


    col = predictions_df.loc[: , "prediction0":"prediction5"]

    predictions_df["avg"] = col.mean(axis=1)

    auc_true = roc_auc_score(test[outcome], predictions_df["avg"])
    AUPRC_true = average_precision_score(test[outcome],predictions_df["avg"])


    test= test.assign(prediction= predictions_df["avg"].values)

    # size = int(len(test) * 1.0)

    for i in range(n_iterations):

        test_resampled = test.sample(frac= 1, replace=True)

        auc = roc_auc_score(test_resampled[outcome], test_resampled["prediction"])
        Test_AUC.append(auc)
        avg = average_precision_score(test_resampled[outcome], test_resampled["prediction"])
        Test_AUPRC.append(avg)

    return(Test_AUPRC, Test_AUC, auc_true, AUPRC_true, predictions_df["avg"], predictions_df)


def compute_Confidence_intervals(list_,true_value):
    """This function calcualted the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    print(f"CI 95% {round(true_value, 3)} ( {round(lower, 3)} , {round(upper, 3)} )")

    return(upper,lower)


def apply_system(outcome, test, selected_models, train_columns):

    """ This function calls the prediction and Confidence interval functions and returns the performance results."""
    testt = test

    Test_AUPRC, Test_AUC, auc_true, AUPRC_true, prediction, _ = predict_with_bootstraps(testt, outcome, selected_models, train_columns)


    print("AUPRC")
    upperAUPRC, lowerAUPRC = compute_Confidence_intervals(Test_AUPRC, AUPRC_true)

    print("AUC ")
    upperAUC, lowerAUC= compute_Confidence_intervals(Test_AUC, auc_true)


    return(auc_true, AUPRC_true, prediction, upperAUC, lowerAUC, upperAUPRC, lowerAUPRC)


def display_top_features(feat, complication):
    for i in range(len(feat)):
        print(f"complication: {complication} \n ------ \n Top 4 predictive features: {feat[i]} \n ----------------")
        

def get_feature_importances(models, test_data, complication, train_columns):
    """ This function calculates the shap values of the top models for each of the investigated complication"""
   
    tree_explain = [LGBMClassifier]

    features = []
    avg_shap_values = []

    test_data_i=  test_data[complication]
    importance_dfs={}
    counter =0

    for x in models[complication]:
        # Get feature importances
        X_importance = test_data_i[train_columns]
        if (type(x) in tree_explain):
            explainer =  shap.TreeExplainer(x)
            shap_values = explainer.shap_values(X_importance,check_additivity=False)

        else:
            model_results = x.predict_proba
            X_importance = X_importance.fillna(X_importance.median())
            X_importance_= shap.sample(X_importance, 50)
            explainer = shap.KernelExplainer(model_results,X_importance_)
            shap_values = explainer.shap_values(X_importance_,check_additivity=False)[0]
        values= np.abs(shap_values).mean(0)

        importance_df = pd.DataFrame()
        importance_df['column_name']= train_columns
        importance_df["importance"]= values
        importance_df= importance_df.sort_values('column_name')
        importance_dfs[counter] = importance_df
        counter = counter+1


    importance_df = pd.DataFrame([X_importance.columns.tolist()]).T
    importance_df.columns = ['column_name']
    for x in range(6):
        importance_df[f"importance_{str(x)}"] =importance_dfs[x]["importance"]

    col = importance_df.loc[: , "importance_0":"importance_6"]

    importance_df["avg"] = col.mean(axis=1)

    features.append(importance_df.sort_values(by="avg", ascending=False).column_name[:4].values)
    avg_shap_values.append(importance_df.sort_values(by="avg", ascending=False).avg[:4].values)

    display_top_features(features, complication)

    return(features, avg_shap_values)


def get_confidence_calibration(true, predict,bins):
    """ This function calcualtes the calibration confidence intervals based on the predictions"""

    slopes = []
    intercepts =[]
    df = pd.DataFrame()
    df["true"] = true.values
    df["pred"] = predict.values

    fpr1, tpr1 = calibration_curve(true, predict, n_bins=bins)
    slope_true, intercept_true, _, _, _ = stats.linregress(tpr1, fpr1)

    for i in range(1000):
        df_resampled = df.sample(frac=1, replace=True)
        fpr1, tpr1 = calibration_curve(df_resampled["true"], df_resampled["pred"], n_bins=bins)
        slope, intercept, _, _, _ = stats.linregress(tpr1, fpr1)
        slopes.append(slope)
        intercepts.append(intercept)
    print("slopes")
    compute_Confidence_intervals(slopes, slope_true)
    print("intercepts")
    compute_Confidence_intervals(intercepts, intercept_true)

    return(slope_true, intercept_true)
