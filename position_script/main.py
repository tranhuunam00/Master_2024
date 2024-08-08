from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import warnings
import array as arr
import shap
import os
from util import create_training_data, calculate_accelerometer_features, calculate_accelerometer_fft_features
from model import LRmodel, RFmodel
import joblib
import glob
import os
import cv2

# Change the current directory to the directory where the code is saved
os.chdir('../input')
np.bool = np.bool_
np.int = np.int_


data = pd.read_csv('threepeople.csv')

data['createdAt'] = data['createdAt'].str[:-38]

print(data.head(5))

x_list, y_list, z_list, train_labels = create_training_data(
    data=data, window_size=20, step_size=10)

features = calculate_accelerometer_features(
    x_list=x_list, y_list=y_list, z_list=z_list, window_size=20)
features_fft = calculate_accelerometer_fft_features(
    x_list=x_list, y_list=y_list, z_list=z_list, window_size=20)
features = features[['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_aad',
                     'y_aad', 'z_aad', 'y_median', 'z_median', 'x_mad', 'y_mad', 'z_mad', 'x_IQR', 'y_IQR',
                     'z_IQR', 'x_neg_count', 'y_neg_count', 'z_neg_count', 'x_pos_count',
                     'y_pos_count', 'z_pos_count', 'x_above_mean', 'y_above_mean', 'z_above_mean', 'x_peak_count', 'y_peak_count', 'z_peak_count',
                     'x_skewness', 'y_skewness', 'z_skewness', 'x_kurtosis', 'y_kurtosis',
                     'z_kurtosis', 'x_energy', 'y_energy', 'z_energy', 'avg_result_accl',
                     'sma']]

features_fft = features_fft[['x_mean_fft', 'y_mean_fft', 'z_mean_fft', 'x_min_fft',
                             'y_min_fft', 'z_min_fft', 'x_maxmin_diff_fft', 'y_maxmin_diff_fft', 'z_maxmin_diff_fft',
                             'x_median_fft', 'y_median_fft', 'z_median_fft', 'x_IQR_fft', 'y_IQR_fft', 'z_IQR_fft', 'x_above_mean_fft', 'y_above_mean_fft', 'z_above_mean_fft',
                             'x_peak_count_fft', 'y_peak_count_fft', 'z_peak_count_fft',
                             'x_kurtosis_fft', 'y_kurtosis_fft', 'z_kurtosis_fft', 'x_energy_fft', 'y_energy_fft',
                             'z_energy_fft', 'avg_result_accl_fft', 'sma_fft']]

features_all = pd.concat([features, features_fft], axis=1)
train, test, labelTrain, labelTest = train_test_split(
    features_all, train_labels, test_size=0.25, random_state=1)

# lr = LRmodel(train=train, test=test,
#              labelTrain=labelTrain, labelTest=labelTest)
rfc = RFmodel(train=train, test=test,
              labelTrain=labelTrain, labelTest=labelTest)
joblib.dump(rfc, 'models.dat')
# explainer = shap.Explainer(rfc.predict, train)
# # Calculates the SHAP values - It takes some time
# shap_values = explainer(train)
# # Evaluate SHAP values
# shap.plots.bar(shap_values, max_display=50)
