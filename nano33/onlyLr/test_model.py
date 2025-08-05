import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from util import (
    create_training_data,
    calculate_accelerometer_features,
    matrixConf
)

# ==== Config ====
INPUT_CSV = './input_nano_33/test4.csv'

FEATURE_COLUMNS = [
    "z_median", "z_mean", "x_mean", "x_energy", "z_pos_count",
    "z_neg_count", "x_median", "z_energy", "avg_result_accl",
    "x_neg_count", "z_std", "x_pos_count", "y_energy", "y_mean",
    "sma", "y_median"
]


def load_and_prepare_data(csv_path):
    print("[INFO] Loading data...")
    df = pd.read_csv(csv_path)

    print("[INFO] Creating training data...")
    x_list, y_list, z_list, labels = create_training_data(
        data=df, window_size=20, step_size=10)

    print("[INFO] Extracting features...")
    features = calculate_accelerometer_features(
        x_list, y_list, z_list, window_size=20)

    features = features[FEATURE_COLUMNS]

    return features, labels


if __name__ == "__main__":
    # Load và xử lý dữ liệu
    features, labels = load_and_prepare_data(INPUT_CSV)
    print(f"[INFO] Total samples: {len(features)}")
    print(f"[INFO] Label distribution: {np.bincount(labels)}")

    # Load model và scaler đã huấn luyện
    print("[INFO] Loading trained model and scaler...")
    model = joblib.load('onlyLR_minmax.dat')
    scaler = joblib.load('onlyScaler_minmax.dat')

    # Chuẩn hóa đặc trưng
    features_scaled = scaler.transform(features)

    # Dự đoán
    predictions = model.predict(features_scaled)

    # In tiêu đề
    print("label,pred")

    # In từng dòng nhãn thực tế và nhãn dự đoán
    for t, p in zip(labels, predictions):
        print(f"{t},{p}")
