from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
import numpy as np


def create_training_data(data, window_size=20, step_size=10):
    """
    Chia dữ liệu cảm biến thành các cửa sổ trượt (sliding windows) 
    và tạo nhãn cho mỗi cửa sổ.
    """
    x_list, y_list, z_list, train_labels = [], [], [], []

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i: i + window_size]

        # Bỏ qua nếu nhãn trong cửa sổ thay đổi
        if window['activity'].nunique() > 1:
            continue

        label = window['activity'].iloc[0]
        if pd.isna(label):
            continue

        x_list.append(window['x'].values)
        y_list.append(window['y'].values)
        z_list.append(window['z'].values)
        train_labels.append(label)

    return x_list, y_list, z_list, train_labels


def calculate_accelerometer_features(x_list, y_list, z_list, window_size=20):
    X_train = pd.DataFrame()

    # mean
    X_train['x_mean'] = [np.mean(x) for x in x_list]
    X_train['y_mean'] = [np.mean(y) for y in y_list]
    X_train['z_mean'] = [np.mean(z) for z in z_list]

    # std dev
    X_train['x_std'] = [np.std(x) for x in x_list]
    X_train['y_std'] = [np.std(y) for y in y_list]
    X_train['z_std'] = [np.std(z) for z in z_list]

    # average absolute deviation
    X_train['x_aad'] = [np.mean(np.abs(x - np.mean(x))) for x in x_list]
    X_train['y_aad'] = [np.mean(np.abs(y - np.mean(y))) for y in y_list]
    X_train['z_aad'] = [np.mean(np.abs(z - np.mean(z))) for z in z_list]

    # min / max
    X_train['x_min'] = [np.min(x) for x in x_list]
    X_train['y_min'] = [np.min(y) for y in y_list]
    X_train['z_min'] = [np.min(z) for z in z_list]

    X_train['x_max'] = [np.max(x) for x in x_list]
    X_train['y_max'] = [np.max(y) for y in y_list]
    X_train['z_max'] = [np.max(z) for z in z_list]

    # max-min difference
    X_train['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
    X_train['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
    X_train['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

    # median / MAD
    X_train['x_median'] = [np.median(x) for x in x_list]
    X_train['y_median'] = [np.median(y) for y in y_list]
    X_train['z_median'] = [np.median(z) for z in z_list]

    X_train['x_mad'] = [np.median(np.abs(x - np.median(x))) for x in x_list]
    X_train['y_mad'] = [np.median(np.abs(y - np.median(y))) for y in y_list]
    X_train['z_mad'] = [np.median(np.abs(z - np.median(z))) for z in z_list]

    # IQR
    X_train['x_IQR'] = [np.percentile(
        x, 75) - np.percentile(x, 25) for x in x_list]
    X_train['y_IQR'] = [np.percentile(
        y, 75) - np.percentile(y, 25) for y in y_list]
    X_train['z_IQR'] = [np.percentile(
        z, 75) - np.percentile(z, 25) for z in z_list]

    # negative / positive counts
    X_train['x_neg_count'] = [np.sum(x < 0) for x in x_list]
    X_train['y_neg_count'] = [np.sum(y < 0) for y in y_list]
    X_train['z_neg_count'] = [np.sum(z < 0) for z in z_list]

    X_train['x_pos_count'] = [np.sum(x > 0) for x in x_list]
    X_train['y_pos_count'] = [np.sum(y > 0) for y in y_list]
    X_train['z_pos_count'] = [np.sum(z > 0) for z in z_list]

    # values above mean
    X_train['x_above_mean'] = [np.sum(x > np.mean(x)) for x in x_list]
    X_train['y_above_mean'] = [np.sum(y > np.mean(y)) for y in y_list]
    X_train['z_above_mean'] = [np.sum(z > np.mean(z)) for z in z_list]

    # number of peaks
    X_train['x_peak_count'] = [len(find_peaks(x)[0]) for x in x_list]
    X_train['y_peak_count'] = [len(find_peaks(y)[0]) for y in y_list]
    X_train['z_peak_count'] = [len(find_peaks(z)[0]) for z in z_list]

    # skewness / kurtosis
    X_train['x_skewness'] = [stats.skew(x) for x in x_list]
    X_train['y_skewness'] = [stats.skew(y) for y in y_list]
    X_train['z_skewness'] = [stats.skew(z) for z in z_list]

    X_train['x_kurtosis'] = [stats.kurtosis(x) for x in x_list]
    X_train['y_kurtosis'] = [stats.kurtosis(y) for y in y_list]
    X_train['z_kurtosis'] = [stats.kurtosis(z) for z in z_list]

    # energy
    X_train['x_energy'] = [np.sum(x**2) / window_size for x in x_list]
    X_train['y_energy'] = [np.sum(y**2) / window_size for y in y_list]
    X_train['z_energy'] = [np.sum(z**2) / window_size for z in z_list]

    # average resultant acceleration
    X_train['avg_result_accl'] = [
        np.mean(np.sqrt(x**2 + y**2 + z**2)) for x, y, z in zip(x_list, y_list, z_list)]

    # signal magnitude area (SMA)
    X_train['sma'] = [(np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / window_size
                      for x, y, z in zip(x_list, y_list, z_list)]

    return X_train


def calculate_accelerometer_fft_features(x_list, y_list, z_list, window_size=20, keep_dc=False):
    """
    Tính 29 đặc trưng miền tần số từ các cửa sổ x_list/y_list/z_list (F1 features)
    - Sử dụng rfft (FFT thực) để lấy n_freq = window_size//2 + 1 hệ số.
    - Loại bỏ thành phần DC nếu keep_dc=False.
    - Trả về DataFrame gồm 29 cột đúng chuẩn: mean, min, maxmin_diff, median, IQR,
      above_mean, peak_count, kurtosis, energy, avg_result_accl_fft, sma_fft.
    """

    nfft = window_size
    n_bins = nfft // 2 + 1

    # FFT magnitude cho mỗi trục
    x_fft_list = [np.abs(np.fft.rfft(x, n=nfft)) for x in x_list]
    y_fft_list = [np.abs(np.fft.rfft(y, n=nfft)) for y in y_list]
    z_fft_list = [np.abs(np.fft.rfft(z, n=nfft)) for z in z_list]

    # Bỏ DC component nếu cần
    if not keep_dc:
        x_fft_list = [a[1:] for a in x_fft_list]
        y_fft_list = [a[1:] for a in y_fft_list]
        z_fft_list = [a[1:] for a in z_fft_list]
        nb = n_bins - 1
    else:
        nb = n_bins

    # ======= BẮT ĐẦU TÍNH 29 ĐẶC TRƯNG =======
    features = {
        # 1–3. Mean FFT
        'x_mean_fft': [np.mean(a) for a in x_fft_list],
        'y_mean_fft': [np.mean(a) for a in y_fft_list],
        'z_mean_fft': [np.mean(a) for a in z_fft_list],

        # 4–6. Min FFT
        'x_min_fft': [np.min(a) for a in x_fft_list],
        'y_min_fft': [np.min(a) for a in y_fft_list],
        'z_min_fft': [np.min(a) for a in z_fft_list],

        # 7–9. Max–Min Difference FFT
        'x_maxmin_diff_fft': [np.max(a) - np.min(a) for a in x_fft_list],
        'y_maxmin_diff_fft': [np.max(a) - np.min(a) for a in y_fft_list],
        'z_maxmin_diff_fft': [np.max(a) - np.min(a) for a in z_fft_list],

        # 10–12. Median FFT
        'x_median_fft': [np.median(a) for a in x_fft_list],
        'y_median_fft': [np.median(a) for a in y_fft_list],
        'z_median_fft': [np.median(a) for a in z_fft_list],

        # 13–15. IQR FFT
        'x_IQR_fft': [np.percentile(a, 75) - np.percentile(a, 25) for a in x_fft_list],
        'y_IQR_fft': [np.percentile(a, 75) - np.percentile(a, 25) for a in y_fft_list],
        'z_IQR_fft': [np.percentile(a, 75) - np.percentile(a, 25) for a in z_fft_list],

        # 16–18. Above mean count FFT
        'x_above_mean_fft': [np.sum(a > np.mean(a)) for a in x_fft_list],
        'y_above_mean_fft': [np.sum(a > np.mean(a)) for a in y_fft_list],
        'z_above_mean_fft': [np.sum(a > np.mean(a)) for a in z_fft_list],

        # 19–21. Peak count FFT
        'x_peak_count_fft': [len(find_peaks(a)[0]) for a in x_fft_list],
        'y_peak_count_fft': [len(find_peaks(a)[0]) for a in y_fft_list],
        'z_peak_count_fft': [len(find_peaks(a)[0]) for a in z_fft_list],

        # 22–24. Kurtosis FFT
        'x_kurtosis_fft': [stats.kurtosis(a) for a in x_fft_list],
        'y_kurtosis_fft': [stats.kurtosis(a) for a in y_fft_list],
        'z_kurtosis_fft': [stats.kurtosis(a) for a in z_fft_list],

        # 25–27. Energy FFT (chuẩn hóa theo nb)
        'x_energy_fft': [np.sum(a**2) / float(nb) for a in x_fft_list],
        'y_energy_fft': [np.sum(a**2) / float(nb) for a in y_fft_list],
        'z_energy_fft': [np.sum(a**2) / float(nb) for a in z_fft_list],
    }

    # 28. Average resultant FFT (mean of sqrt(x^2+y^2+z^2))
    features['avg_result_accl_fft'] = [
        np.mean(np.sqrt(xf**2 + yf**2 + zf**2))
        for xf, yf, zf in zip(x_fft_list, y_fft_list, z_fft_list)
    ]

    # 29. Signal Magnitude Area (SMA)
    features['sma_fft'] = [
        (np.sum(np.abs(xf)) + np.sum(np.abs(yf)) + np.sum(np.abs(zf))) / float(nb)
        for xf, yf, zf in zip(x_fft_list, y_fft_list, z_fft_list)
    ]

    # ======= HOÀN THIỆN =======
    features_fft = pd.DataFrame(features)
    features_fft['n_fft_bins'] = nb  # thêm để debug, có thể bỏ

    return features_fft


def create_training_data_NN_like_micro(data, window_size=10, step_size=5):
    """
    Tạo dữ liệu dạng interleaved giống như tflInputTensor của thiết bị:
    [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
    """
    total_list_NN = []
    train_labels_NN = []

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i: i + window_size]

        # Bỏ qua nếu trong cửa sổ có nhiều nhãn
        if window['activity'].nunique() > 1:
            continue

        # Lấy mảng x,y,z
        x = window['x'].values
        y = window['y'].values
        z = window['z'].values

        # Tạo mảng xen kẽ: x1,y1,z1,x2,y2,z2,...
        interleaved = np.empty(window_size * 3, dtype=np.float32)
        interleaved[0::3] = x
        interleaved[1::3] = y
        interleaved[2::3] = z

        total_list_NN.append(interleaved)
        train_labels_NN.append(window['activity'].iloc[0])

    print(
        f"Created {len(total_list_NN)} windows × {window_size} samples (interleaved format)")
    return np.array(total_list_NN), np.array(train_labels_NN)
