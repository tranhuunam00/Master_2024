import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
import math
import array as arr
import numpy as np
np.bool = np.bool_
np.int = np.int_


def create_training_data(data, window_size, step_size):
    """Creates training data for a machine learning model.

    Args:
      data: A Pandas DataFrame containing the sensor data.
      window_size: The size of the window to use for each training example.
      step_size: The step size to use when sliding the window across the data.

    Returns:
      A list of training examples and a list of corresponding labels.
    """

    x_list = []
    y_list = []
    z_list = []
    train_labels = []

    for i in range(0, data.shape[0] - window_size, step_size):
        xs = data['x'].values[i: i + window_size]
        ys = data['y'].values[i: i + window_size]
        zs = data['z'].values[i: i + window_size]

        # Skip examples where the activity label changes within the window.
        if (data['activity'][i+1] != data['activity'][i + window_size-2]):
            print(i)
            continue

        label = data['activity'][i]

        # Skip examples where the label is NaN.
        if math.isnan(label):
            continue

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        train_labels.append(label)

    return x_list, y_list, z_list, train_labels


def create_training_data_NN(data):
    print("STARTING create_training_data_NN")
    total_list_NN = []
    train_labels_NN = []

    for i in range(0, data.shape[0] - 1, 1):
        x = data['x'].values[i]
        y = data['y'].values[i]
        z = data['z'].values[i]
        total_list_NN.append([x, y, z])
        label = data['activity'][i]
        train_labels_NN.append(label)

    return total_list_NN, train_labels_NN


def calculate_accelerometer_features(x_list, y_list, z_list, window_size=20):
    X_train = pd.DataFrame()

    # mean
    X_train['x_mean'] = pd.Series(x_list).apply(lambda x: x.mean())
    X_train['y_mean'] = pd.Series(y_list).apply(lambda x: x.mean())
    X_train['z_mean'] = pd.Series(z_list).apply(lambda x: x.mean())

    # std dev
    X_train['x_std'] = pd.Series(x_list).apply(lambda x: x.std())
    X_train['y_std'] = pd.Series(y_list).apply(lambda x: x.std())
    X_train['z_std'] = pd.Series(z_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['x_aad'] = pd.Series(x_list).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_aad'] = pd.Series(y_list).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_aad'] = pd.Series(z_list).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['x_min'] = pd.Series(x_list).apply(lambda x: x.min())
    X_train['y_min'] = pd.Series(y_list).apply(lambda x: x.min())
    X_train['z_min'] = pd.Series(z_list).apply(lambda x: x.min())

    # max
    X_train['x_max'] = pd.Series(x_list).apply(lambda x: x.max())
    X_train['y_max'] = pd.Series(y_list).apply(lambda x: x.max())
    X_train['z_max'] = pd.Series(z_list).apply(lambda x: x.max())

    # max-min diff
    X_train['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
    X_train['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
    X_train['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

    # median
    X_train['x_median'] = pd.Series(x_list).apply(lambda x: np.median(x))
    X_train['y_median'] = pd.Series(y_list).apply(lambda x: np.median(x))
    X_train['z_median'] = pd.Series(z_list).apply(lambda x: np.median(x))

    # Mean Absolute Deviation" (Độ lệch tuyệt đối trung bình)
    X_train['x_mad'] = pd.Series(x_list).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_mad'] = pd.Series(y_list).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_mad'] = pd.Series(z_list).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range Interquartile Range" (Phạm vi tứ phân vị) trong thống kê.
    X_train['x_IQR'] = pd.Series(x_list).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_IQR'] = pd.Series(y_list).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_IQR'] = pd.Series(z_list).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # negative count
    X_train['x_neg_count'] = pd.Series(x_list).apply(lambda x: np.sum(x < 0))
    X_train['y_neg_count'] = pd.Series(y_list).apply(lambda x: np.sum(x < 0))
    X_train['z_neg_count'] = pd.Series(z_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['x_pos_count'] = pd.Series(x_list).apply(lambda x: np.sum(x > 0))
    X_train['y_pos_count'] = pd.Series(y_list).apply(lambda x: np.sum(x > 0))
    X_train['z_pos_count'] = pd.Series(z_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['x_above_mean'] = pd.Series(
        x_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['y_above_mean'] = pd.Series(
        y_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['z_above_mean'] = pd.Series(
        z_list).apply(lambda x: np.sum(x > x.mean()))

    # number of peaks (số lượng đỉnh) trong một tập dữ liệu số liệu 1 chiều
    X_train['x_peak_count'] = pd.Series(
        x_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_peak_count'] = pd.Series(
        y_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_peak_count'] = pd.Series(
        z_list).apply(lambda x: len(find_peaks(x)[0]))

    # skewness
    X_train['x_skewness'] = pd.Series(x_list).apply(lambda x: stats.skew(x))
    X_train['y_skewness'] = pd.Series(y_list).apply(lambda x: stats.skew(x))
    X_train['z_skewness'] = pd.Series(z_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['x_kurtosis'] = pd.Series(
        x_list).apply(lambda x: stats.kurtosis(x))
    X_train['y_kurtosis'] = pd.Series(
        y_list).apply(lambda x: stats.kurtosis(x))
    X_train['z_kurtosis'] = pd.Series(
        z_list).apply(lambda x: stats.kurtosis(x))

    # energy
    X_train['x_energy'] = pd.Series(x_list).apply(
        lambda x: np.sum(x**2)/window_size)
    X_train['y_energy'] = pd.Series(y_list).apply(
        lambda x: np.sum(x**2)/window_size)
    X_train['z_energy'] = pd.Series(z_list).apply(
        lambda x: np.sum(x**2/window_size))

    # avg resultant
    X_train['avg_result_accl'] = [i.mean() for i in (
        (pd.Series(x_list)**2 + pd.Series(y_list)**2 + pd.Series(z_list)**2)**0.5)]

    # signal magnitude area
    X_train['sma'] = pd.Series(x_list).apply(lambda x: np.sum(abs(x)/window_size)) + pd.Series(y_list).apply(
        lambda x: np.sum(abs(x)/window_size)) + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/window_size))
    return X_train


def calculate_accelerometer_fft_features(x_list, y_list, z_list, window_size=20):
    x_list_fft = pd.Series(x_list).apply(
        lambda x: np.abs(np.fft.fft(x))[1:window_size+1])
    y_list_fft = pd.Series(y_list).apply(
        lambda x: np.abs(np.fft.fft(x))[1:window_size+1])
    z_list_fft = pd.Series(z_list).apply(
        lambda x: np.abs(np.fft.fft(x))[1:window_size+1])

    features_fft = pd.DataFrame()
    # Statistical Features on raw x, y and z in frequency domain
    # FFT mean
    features_fft['x_mean_fft'] = pd.Series(
        x_list_fft).apply(lambda x: x.mean())
    features_fft['y_mean_fft'] = pd.Series(
        y_list_fft).apply(lambda x: x.mean())
    features_fft['z_mean_fft'] = pd.Series(
        z_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    features_fft['x_std_fft'] = pd.Series(x_list_fft).apply(lambda x: x.std())
    features_fft['y_std_fft'] = pd.Series(y_list_fft).apply(lambda x: x.std())
    features_fft['z_std_fft'] = pd.Series(z_list_fft).apply(lambda x: x.std())

    # FFT avg absolute diff
    features_fft['x_aad_fft'] = pd.Series(x_list_fft).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))
    features_fft['y_aad_fft'] = pd.Series(y_list_fft).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))
    features_fft['z_aad_fft'] = pd.Series(z_list_fft).apply(
        lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    features_fft['x_min_fft'] = pd.Series(x_list_fft).apply(lambda x: x.min())
    features_fft['y_min_fft'] = pd.Series(y_list_fft).apply(lambda x: x.min())
    features_fft['z_min_fft'] = pd.Series(z_list_fft).apply(lambda x: x.min())

    # FFT max
    features_fft['x_max_fft'] = pd.Series(x_list_fft).apply(lambda x: x.max())
    features_fft['y_max_fft'] = pd.Series(y_list_fft).apply(lambda x: x.max())
    features_fft['z_max_fft'] = pd.Series(z_list_fft).apply(lambda x: x.max())

    # FFT max-min diff
    features_fft['x_maxmin_diff_fft'] = features_fft['x_max_fft'] - \
        features_fft['x_min_fft']
    features_fft['y_maxmin_diff_fft'] = features_fft['y_max_fft'] - \
        features_fft['y_min_fft']
    features_fft['z_maxmin_diff_fft'] = features_fft['z_max_fft'] - \
        features_fft['z_min_fft']

    # FFT median
    features_fft['x_median_fft'] = pd.Series(
        x_list_fft).apply(lambda x: np.median(x))
    features_fft['y_median_fft'] = pd.Series(
        y_list_fft).apply(lambda x: np.median(x))
    features_fft['z_median_fft'] = pd.Series(
        z_list_fft).apply(lambda x: np.median(x))

    # FFT median abs dev
    features_fft['x_mad_fft'] = pd.Series(x_list_fft).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))
    features_fft['y_mad_fft'] = pd.Series(y_list_fft).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))
    features_fft['z_mad_fft'] = pd.Series(z_list_fft).apply(
        lambda x: np.median(np.absolute(x - np.median(x))))

    # FFT Interquartile range
    features_fft['x_IQR_fft'] = pd.Series(x_list_fft).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    features_fft['y_IQR_fft'] = pd.Series(y_list_fft).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    features_fft['z_IQR_fft'] = pd.Series(z_list_fft).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT values above mean
    features_fft['x_above_mean_fft'] = pd.Series(
        x_list_fft).apply(lambda x: np.sum(x > x.mean()))
    features_fft['y_above_mean_fft'] = pd.Series(
        y_list_fft).apply(lambda x: np.sum(x > x.mean()))
    features_fft['z_above_mean_fft'] = pd.Series(
        z_list_fft).apply(lambda x: np.sum(x > x.mean()))

    # FFT number of peaks
    features_fft['x_peak_count_fft'] = pd.Series(
        x_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    features_fft['y_peak_count_fft'] = pd.Series(
        y_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    features_fft['z_peak_count_fft'] = pd.Series(
        z_list_fft).apply(lambda x: len(find_peaks(x)[0]))

    # FFT skewness
    features_fft['x_skewness_fft'] = pd.Series(
        x_list_fft).apply(lambda x: stats.skew(x))
    features_fft['y_skewness_fft'] = pd.Series(
        y_list_fft).apply(lambda x: stats.skew(x))
    features_fft['z_skewness_fft'] = pd.Series(
        z_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    features_fft['x_kurtosis_fft'] = pd.Series(
        x_list_fft).apply(lambda x: stats.kurtosis(x))
    features_fft['y_kurtosis_fft'] = pd.Series(
        y_list_fft).apply(lambda x: stats.kurtosis(x))
    features_fft['z_kurtosis_fft'] = pd.Series(
        z_list_fft).apply(lambda x: stats.kurtosis(x))

    # FFT energy
    features_fft['x_energy_fft'] = pd.Series(
        x_list_fft).apply(lambda x: np.sum(x**2)/20)
    features_fft['y_energy_fft'] = pd.Series(
        y_list_fft).apply(lambda x: np.sum(x**2)/20)
    features_fft['z_energy_fft'] = pd.Series(
        z_list_fft).apply(lambda x: np.sum(x**2/20))

    # FFT avg resultant
    features_fft['avg_result_accl_fft'] = [i.mean() for i in ((pd.Series(
        x_list_fft)**2 + pd.Series(y_list_fft)**2 + pd.Series(z_list_fft)**2)**0.5)]

    # FFT Signal magnitude area
    features_fft['sma_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(abs(x)/window_size)) + pd.Series(y_list_fft).apply(
        lambda x: np.sum(abs(x)/window_size)) + pd.Series(z_list_fft).apply(lambda x: np.sum(abs(x)/window_size))

    return features_fft


def matrixConf(y_test, y_pred):
    labels = ["supine", "left side", "right side", "prone"]
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                annot=True, linewidths=0.1, fmt="d", cmap="YlGnBu")
    plt.title("Confusion matrix", fontsize=15)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def getCorr(data, fields):
    import seaborn as sns
    tran_corr = data[fields].corr()
    print("getCorr", tran_corr)
    plt.figure(figsize=(10, 10))
    sns.heatmap(tran_corr, annot=True, cmap='coolwarm',)
    plt.title('Correlation Heatmap')
    plt.show()
    return tran_corr
    
