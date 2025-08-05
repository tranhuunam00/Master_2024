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
INPUT_CSV = './input_nano_33/total.csv'

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


def train_LR_model(X_train, X_test, y_train, y_test):
    print("[INFO] Scaling features with MinMaxScaler [-1, 1]...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[INFO] Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=21,
        max_iter=100,
        multi_class="ovr",
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    matrixConf(y_test, y_pred)

    return model, scaler


def export_model_parameters(model, scaler):
    print("\n===== Model Parameters (for Arduino) =====")
    np.set_printoptions(suppress=True, precision=6)

    print("\n// Coefficients (weights):")
    print(model.coef_)

    print("\n// Intercepts (biases):")
    print(model.intercept_)

    print("\n// Scaler min:")
    print(scaler.data_min_)

    print("\n// Scaler max:")
    print(scaler.data_max_)

    print("\n// Scaler scale:")
    print(scaler.scale_)  # scale = 2 / (max - min)


if __name__ == "__main__":
    features, labels = load_and_prepare_data(INPUT_CSV)

    print(f"[INFO] Total samples: {len(features)}")
    print(f"[INFO] Label distribution: {np.bincount(labels)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=1
    )

    print(f"[INFO] Train label distribution: {np.bincount(y_train)}")
    print(f"[INFO] Test label distribution: {np.bincount(y_test)}")

    # Train model
    model, scaler = train_LR_model(X_train, X_test, y_train, y_test)

    # Save model
    joblib.dump(model, 'onlyLR_minmax.dat')
    joblib.dump(scaler, 'onlyScaler_minmax.dat')
    print("[INFO] Model saved to onlyLR_minmax.dat and onlyScaler_minmax.dat")

    # Export weights for Arduino
    export_model_parameters(model, scaler)
