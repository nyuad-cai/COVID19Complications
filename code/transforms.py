from sklearn import preprocessing

def get_transforms_MinMax_Scaler(training,train_columns):
    """ Get MinMax scalar transforms"""
    feat = train_columns
    transforms = {}
    for m in feat:
        scaler_mean =  preprocessing.MinMaxScaler(feature_range=(0,1))
        train_minmax =scaler_mean.fit_transform(training[[m]].dropna())
        transforms[m] = scaler_mean
    return(transforms,feat,train_minmax)

def apply_transforms_MinMax_Scaler(dataset,training,train_columns):
    """ Apply MinMax scalar transforms """
    transforms,feat,train_minmax = get_transforms_MinMax_Scaler(training,train_columns)
    dataset_copy = dataset.copy()
    for m in feat:
        func = [value for key, value in transforms.items() if key in m][0]
        dataset_copy[m] = func.transform(dataset_copy[m].values.reshape(len(dataset_copy), 1))
    return dataset_copy,train_minmax

def get_transforms_STD_Scaler(training,train_columns):
    """ Get Standard scalar transforms"""
    feat = train_columns
    transforms = {}
    for m in feat:
        scaler_mean =  preprocessing.StandardScaler(feature_range=(0,1))
        _=scaler_mean.fit_transform(training[[m]].dropna())
        transforms[m] = scaler_mean
    return(transforms,feat)

def apply_transforms_STD_Scaler(dataset,training,train_columns):
    """ Apply Standard scalar transforms """

    transforms,feat = get_transforms_STD_Scaler(training,train_columns)
    dataset_copy = dataset.copy()
    for m in feat:
        func = [value for key, value in transforms.items() if key in m][0]
        dataset_copy[m] = func.transform(dataset_copy[m].values.reshape(len(dataset_copy), 1))
    return dataset_copy