import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
#loading useful libraries
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from transforms import *
from model import *
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def search_models(training_i, x, train_columns):

        """
        This function computes the performance using 5 baseline models. 
        It was done through stratified k-folds cross-validation using the complication's respective training set, with k= 3. 
        We performed random hyperparameter search for all the hyperparameters over 20 iterations. 
        We finally selected the top two set of hyperparameters that achieved the highest average area under the receiving operator characteristic curve (AUROC) on the validation sets, 
        resulting with 6 final models per complication. The function below returns the 6 models for each respective training subset.
        """

        baselines = ["Logistic Regression", "KNN", "LGBM", "MLP", "SVM"]
        baselines_need_transforms = ["Logistics Regression", "MLP"]
        baselines_need_transforms2 = ["KNN", "SVM"]

        X = training_i[train_columns]
        Y = training_i[x]

        n_iterations = 20 # number of iterations for random search
        top_n = 2# select top n parameter sets
        all_ = {}

        # prepare indexes for stratified cross validation
        skf = StratifiedKFold(n_splits= 3, shuffle=True, random_state=0)
        skf.get_n_splits(X, Y)

        print ("Stratified K Fold into 3 splits ...")


        for base in (baselines):
            print(f"Search in progress : {base}")

            roc_auc_mean, auprc_mean, dict_list, model_list,train_list,val_list = ([] for i in range(6))
            models_1,vals_1,trains_1 = ([] for i in range(3))


            for i in range(0, n_iterations):
                if ((i+1) % 10 == 0):
                    print (f"Random search {i+1}...")

                skf_split = skf.split(X, Y)
                param_dictionary , model, X_  = choose_baseline(base, X)
                roc_in_k, pr_in_k, clf_in_k, train_in_k, val_in_k = ([] for i in range(5))
                j = 0

                for train_index, val_index in skf_split:
                    X_train = X_.iloc[train_index]
                    y_train = Y.iloc[train_index]


                    X_val = X_.iloc[val_index]
                    y_val = Y.iloc[val_index]
                    if (model in baselines_need_transforms):

                        X_val = apply_transforms_MinMax_Scaler(X_val, X_train, train_columns)
                        X_train = apply_transforms_MinMax_Scaler(X_val, X_train, train_columns)

                    if (model in baselines_need_transforms2):

                        X_val = apply_transforms_STD_Scaler(X_val, X_train, train_columns)
                        X_train = apply_transforms_STD_Scaler(X_val, X_train, train_columns)

                    clf = model(**param_dictionary)

                    clf = clf.fit(X_train, y_train)

                    # predicting
                    y_pred = clf.predict_proba(X_val)[:, 1]

#                     calculate performance across folds
                    roc = roc_auc_score(y_val, y_pred)
                    AUPRC = average_precision_score(y_val, y_pred)
                    pr_in_k.append(AUPRC)
                    roc_in_k.append(roc)
                    roc_array = np.asarray(roc_in_k)
                    pr_array = np.asarray(pr_in_k)
                    clf_in_k.append(clf)
                    train_in_k.append(train_index)
                    val_in_k.append(val_index)
                    j = j+1


            #   append the lists for each hyperparameter search

                roc_auc_mean.append(roc_array.mean())
                auprc_mean.append(pr_array.mean())
                dict_list.append(param_dictionary)
                val_list.append(val_in_k)
                train_list.append(train_in_k)
                model_list.append(clf_in_k)
                gc.collect()

        # Storing results for this model
            print(f"Storing results of top models for {base}")
            results_pd = pd.DataFrame({
                "avg_roc_auc": roc_auc_mean,
                "avg_auprc":auprc_mean,
                "clf_s":model_list,
                "validation_sets":val_list,
                "train_sets": train_list})

            results_pd.sort_values("avg_roc_auc", ascending=False, axis = 0, inplace=True)
            top_pd = results_pd.head(top_n)
            models_1.append(top_pd['clf_s'].values[0:3][:6])
            vals_1.append(top_pd['validation_sets'].values[0:3][:6])
            trains_1.append(top_pd['train_sets'].values[0:3][:6])
            param_df = pd.DataFrame()
            param_df["models"] = models_1
            param_df["vals"] = vals_1
            param_df["trains"] = trains_1
            param_df["auc_val"] = top_pd.avg_roc_auc.mean()
            param_df["auprc_val"] = top_pd.avg_auprc.mean()

            val_sets,train_sets,models_ = ([] for i in range(3))

            for i in range (len(param_df.vals[0])):
                for j in (param_df.vals[0][i]):
                    val_sets.append(j)
            for i in range (len(param_df.trains[0])):
                for j in (param_df.trains[0][i]):
                    train_sets.append(j)

            for i in range (len(param_df.models[0])):
                for j in (param_df.models[0][i]):
                    models_.append(j)

#             storing top models and performance for all the different types of models
            all_[base] = [models_, param_df["auc_val"].values, param_df["auprc_val"].values, val_sets, train_sets]

        return(all_)

def get_models(complications, training_subsets, train_columns):
    """ 
    This function calls  the model selection function
    then calibrates each of the 6 models per complication-specific prediction.
    """
    models_all = {}
    ms = {}
    vals = {}
    trains = {}

    for i in complications:
        print(f"Searching for models to predict: {i}")
        models_all[i] = pd.DataFrame()
        training_i =  training_subsets[i]
        all_= search_models(training_i, (f"{i}_after24"), train_columns)
        chosen_model = get_top_models(all_).reset_index()[:1]
        ms[i] = chosen_model.Models[0]
        vals[i] = chosen_model.Val_sets[0]
        trains[i] = chosen_model.Train_sets[0]
        models_all[i] = calibrate(ms[i],vals[i],training_i,train_columns,i)

        # models_all[i] = clf_calibrated_list

    return(models_all, trains, ms)

def get_top_models(all_):
    """ 
    This function takes the dictionary of all the baselines models and their performances, 
    it then returns a sorted dataframe based on the top AUC
    """
    All_models= pd.DataFrame()
    auc_=[]
    prc_ = []
    name_= []
    vals_ = []
    models_ = []
    trains_=[]
    for i in all_:
        auc_.append(all_[i][1][0])
        prc_.append(all_[i][2][0])
        name_.append(i)
        models_.append(all_[i][0])
        vals_.append(all_[i][3])
        trains_.append(all_[i][4])

    All_models["Baslines"] = name_
    All_models["Avg AUC"] = auc_
    All_models["Avg AUPRC"] = prc_
    All_models["Models"] = models_
    All_models["Val_sets"] = vals_
    All_models["Train_sets"] = trains_
    All_models.sort_values("Avg AUC", ascending = False, axis=0, inplace=True)

    try:
        display(All_models[["Baslines", "Avg AUC", "Avg AUPRC", "Models"]])
    except:
        print(All_models[["Baslines", "Avg AUC", "Avg AUPRC", "Models"]])
    
    return(All_models)

def calibrate(models, validation_sets, training_subset, train_columns, complication):

    """ This function calibrates the selected models for each complication"""
    count = 0
    clf_calibrated_list = []
    clf_calibrated = None

    print("calibrating models...")
    for j, k in zip(models, validation_sets):
            X_val = training_subset[train_columns].iloc[k]
            y_val = training_subset[f"{complication}_after24"].iloc[k]

            if (type(j) == LogisticRegression):
                clf_calibrated = j
            else:
                if (type(j) != LGBMClassifier):
                    X_val = X_val.fillna(X_val.median())
                    X_val = apply_transforms_MinMax_Scaler(X_val, training_subset, train_columns)
                clf_calibrated = CalibratedClassifierCV(j, method = "isotonic", cv="prefit").fit(X_val, y_val)

            
            clf_calibrated_list.append(clf_calibrated)
            count = count + 1

    return(clf_calibrated_list)
