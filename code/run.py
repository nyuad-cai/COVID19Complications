import warnings
warnings.filterwarnings("ignore")
#loading useful libraries
import pandas as pd
from dataset import apply_stratified_framework, get_targets

from plot import plot_PRC, plot_roc
from train import get_models
from test_models import get_results
import os

try: 
    os.mkdir('plots') 
except OSError as error: 
    print(error) 

# please provide the path to your training and testing datasets
training= pd.read_csv("...")
test= pd.read_csv("....")

complications = ['Elevated_troponin', 'Elevated_d-dimer',  'Elevated_Amino','Elevated_IL6', 'SBI', 'AKI', 'ARDS']


framework_train = apply_stratified_framework(training, complications)
framework_test = apply_stratified_framework(test, complications)


targets = get_targets(complications)
train_columns = [x for x in training.columns if x not in targets]

models_all, trainsets, classifers = get_models(complications, framework_train, train_columns )

true_ouctomes, predicted_ouctomes = get_results(framework_test, complications, models_all, train_columns)

plot_roc(complications, true_ouctomes, predicted_ouctomes)

plot_PRC(complications, true_ouctomes, predicted_ouctomes)
