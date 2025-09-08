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
from util import create_training_data, calculate_accelerometer_features, calculate_accelerometer_fft_features, create_training_data_NN, getCorr
from model import LRmodel, RFmodel, SVMmodel, GradientBoostingModel, NeuralNetworkModel2
import joblib
import glob
import os
import cv2

# Change the current directory to the directory where the code is saved
os.chdir('../input/input_kichban')
np.bool = np.bool_
np.int = np.int_


data = pd.read_csv('totalData2.csv')
#  data test
data_test = pd.read_csv('new_nam.csv')

data['createdAt'] = data['createdAt'].str[:-38]
data_test['createdAt'] = data_test['createdAt'].str[:-38]


x_list, y_list, z_list, train_labels = create_training_data(
    data=data, window_size=20, step_size=10)
total_list_NN, train_labels_NN = create_training_data_NN(data=data)

# test
x_list_test, y_list_test, z_list_test, train_labels_test = create_training_data(
    data=data_test, window_size=20, step_size=10)
total_list_NN_test, train_labels_NN_test = create_training_data_NN(
    data=data_test)


features = calculate_accelerometer_features(
    x_list=x_list, y_list=y_list, z_list=z_list, window_size=20)
# test
features_test = calculate_accelerometer_features(
    x_list=x_list_test, y_list=y_list_test, z_list=z_list_test, window_size=20)

print("features", len(features))

# print("features", features[800])


features_fft = calculate_accelerometer_fft_features(
    x_list=x_list, y_list=y_list, z_list=z_list, window_size=10)
# test
features_fft_test = calculate_accelerometer_fft_features(
    x_list=x_list_test, y_list=y_list_test, z_list=z_list_test, window_size=10)


print("features", features_fft.head())


features = features[['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_aad',
                     'y_aad', 'z_aad', 'x_median', 'y_median', 'z_median', 'x_mad', 'y_mad', 'z_mad', 'x_IQR', 'y_IQR',
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

tran_corr = getCorr(features, ['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_aad',
                               'y_aad', 'z_aad', "x_median", 'y_median', 'z_median', 'x_mad', 'y_mad', 'z_mad'])


features = features[['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_aad',
                     'y_aad', 'z_aad', 'x_median', 'y_median', 'z_median', 'x_mad', 'y_mad', 'z_mad', 'x_IQR', 'y_IQR',
                     'z_IQR', 'x_neg_count', 'y_neg_count', 'z_neg_count', 'x_pos_count',
                     'y_pos_count', 'z_pos_count', 'x_above_mean', 'y_above_mean', 'z_above_mean', 'x_peak_count', 'y_peak_count', 'z_peak_count',
                     'x_skewness', 'y_skewness', 'z_skewness', 'x_kurtosis', 'y_kurtosis',
                     'z_kurtosis', 'x_energy', 'y_energy', 'z_energy', 'avg_result_accl',
                     'sma']]

features_test = features_test[['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_aad',
                               'y_aad', 'z_aad', 'x_median', 'y_median', 'z_median', 'x_mad', 'y_mad', 'z_mad', 'x_IQR', 'y_IQR',
                               'z_IQR', 'x_neg_count', 'y_neg_count', 'z_neg_count', 'x_pos_count',
                               'y_pos_count', 'z_pos_count', 'x_above_mean', 'y_above_mean', 'z_above_mean', 'x_peak_count', 'y_peak_count', 'z_peak_count',
                               'x_skewness', 'y_skewness', 'z_skewness', 'x_kurtosis', 'y_kurtosis',
                               'z_kurtosis', 'x_energy', 'y_energy', 'z_energy', 'avg_result_accl',
                               'sma']]

features_fft_test = features_fft_test[['x_mean_fft', 'y_mean_fft', 'z_mean_fft', 'x_min_fft',
                                       'y_min_fft', 'z_min_fft', 'x_maxmin_diff_fft', 'y_maxmin_diff_fft', 'z_maxmin_diff_fft',
                                       'x_median_fft', 'y_median_fft', 'z_median_fft', 'x_IQR_fft', 'y_IQR_fft', 'z_IQR_fft', 'x_above_mean_fft', 'y_above_mean_fft', 'z_above_mean_fft',
                                       'x_peak_count_fft', 'y_peak_count_fft', 'z_peak_count_fft',
                                       'x_kurtosis_fft', 'y_kurtosis_fft', 'z_kurtosis_fft', 'x_energy_fft', 'y_energy_fft',
                                       'z_energy_fft', 'avg_result_accl_fft', 'sma_fft']]

features_all = pd.concat([features, features_fft], axis=1)
features_all_test = pd.concat([features_test, features_fft_test], axis=1)


print("features_all", features_all.shape)
# train, test, labelTrain, labelTest = train_test_split(
#     features_all, train_labels, test_size=0.25, random_state=1)

# trainNN, testNN, labelTrainNN, labelTestNN = train_test_split(
#     total_list_NN, train_labels_NN, test_size=0.25, random_state=1)

train = features_all
test = features_all_test
labelTrain = train_labels
labelTest = train_labels_test
trainNN = total_list_NN
testNN = total_list_NN_test
labelTrainNN = train_labels_NN
labelTestNN = train_labels_NN_test

print(len(trainNN))
print(len(labelTrainNN))

lr = LRmodel(train=train, test=test,
             labelTrain=labelTrain, labelTest=labelTest)
rfc = RFmodel(train=train, test=test,
              labelTrain=labelTrain, labelTest=labelTest)
svm_model = SVMmodel(train=train, test=test,
                     labelTrain=labelTrain, labelTest=labelTest)
gbM = GradientBoostingModel(train=train, test=test,
                            labelTrain=labelTrain, labelTest=labelTest)
nnM = NeuralNetworkModel2(train=trainNN, test=testNN,
                          labelTrain=labelTrainNN, labelTest=labelTestNN)
joblib.dump(rfc, 'models.dat')
explainer = shap.Explainer(rfc.predict, train)
# Calculates the SHAP values - It takes some time
shap_values = explainer(train)
# Evaluate SHAP values
shap.plots.bar(shap_values, max_display=50)
