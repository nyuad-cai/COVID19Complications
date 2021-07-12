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

complications = ['SBI', 'AKI', 'ARDS']


framework_train = apply_stratified_framework(training, complications)
framework_test = apply_stratified_framework(test, complications)


train_columns = ['Diastolic Blood Pressure_mean', 'Diastolic Blood Pressure_min', 'Oxygen Saturation_max', 'Oxygen Saturation_mean', 'Oxygen Saturation_min', 'Peripheral Pulse Rate_max', 'Peripheral Pulse Rate_mean', 'Peripheral Pulse Rate_min', 'Respiratory Rate_max', 'Respiratory Rate_mean', 'Respiratory Rate_min', 'Systolic Blood Pressure_max', 'Systolic Blood Pressure_mean', 'Systolic Blood Pressure_min', 'Temperature Axillary_max', 'Temperature Axillary_mean', 'Temperature Axillary_min', 'GCS_mean', 'GCS_min', 'GCS_max','GENDER','AGE', 'COUGH', 'FEVER', 'SOB', 'SORE_THROAT', 'RASH', 'BMI', 'DIABETES', 'HYPERTENSION', 'CKD', 'CANCER']


models_all, trainsets, classifers = get_models(complications, framework_train, train_columns )

true_ouctomes, predicted_ouctomes = get_results(framework_test, complications, models_all, train_columns)

plot_roc(complications, true_ouctomes, predicted_ouctomes,"testset")

plot_PRC(complications, true_ouctomes, predicted_ouctomes,"testset")