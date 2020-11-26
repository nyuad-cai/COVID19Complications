
def apply_stratified_framework(df, complications):
    """ 
    For each complications-specific model, we excluded from the training set patients who reached that complications prior to the time of prediction
    We also excluded CKD patients for the AKI subsets
    """

    subsets =  dict()
    for i in complications:
         subsets[i] = df[df[f"{i}_before24"] != 1]

    subsets["AKI"] = subsets["AKI"][subsets["AKI"].CKD != 1.0]
    return(subsets)

def get_targets(complications):
    """This function returns the target columns for the investigated complications"""
    targets = []
    for i in complications:
        targets.append(f"{i}_after24")
    return(targets)