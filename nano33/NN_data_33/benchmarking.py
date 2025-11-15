#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import serial
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


SERIAL_PORT = "COM5"
BAUDRATE = 115200
CSV_FILE = "test_div10.csv"
OUT_CSV = "nn_mcu_test_result.csv"

WINDOW_SIZE = 10
STEP_SIZE = 5
SERIAL_TIMEOUT = 2.0
LINE_DELAY = 0.001


# ---------------------------
#  Sử dụng đúng hàm user yêu cầu
# ---------------------------
def create_training_data_NN_like_micro(data, window_size=10, step_size=5):
    """
    Tạo dữ liệu interleaved giống input model TinyML:
    [x1, y1, z1, x2, y2, z2, ...]
    """
    total_list_NN = []
    train_labels_NN = []

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i: i + window_size]

        # Nếu có nhiều activity trong cửa sổ → bỏ
        if window["activity"].nunique() > 1:
            continue

        x = window["x"].values
        y = window["y"].values
        z = window["z"].values

        interleaved = np.empty(window_size * 3, dtype=np.float32)
        interleaved[0::3] = x
        interleaved[1::3] = y
        interleaved[2::3] = z

        total_list_NN.append(interleaved)
        train_labels_NN.append(int(window["activity"].iloc[0]))

    print(f"Created {len(total_list_NN)} windows (interleaved)")
    return np.array(total_list_NN), np.array(train_labels_NN)


# ---------------------------
#  MAIN
# ---------------------------
def main():
    df = pd.read_csv("./input_nano_33/test_div10.csv")

    # Lấy dữ liệu interleaved (y hệt huấn luyện)
    X_interleaved, labels = create_training_data_NN_like_micro(
        df, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )

    print(f"Prepared {len(labels)} windows total")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    preds = []

    try:
        for wi in range(len(labels)):
            w = X_interleaved[wi]   # length = 30 (10×3)

            # gửi xuống MCU dưới dạng từng dòng x,y,z
            for i in range(WINDOW_SIZE):
                x = w[i*3 + 0]
                y = w[i*3 + 1]
                z = w[i*3 + 2]
                ser.write(f"{x},{y},{z}\n".encode())
                time.sleep(LINE_DELAY)

            # nhận lại 1 nhãn
            raw = ser.readline().decode(errors="ignore").strip()

            # trích số nguyên
            candidate = ''.join([c for c in raw if c.isdigit()])
            if candidate == "":
                preds.append(-1)
            else:
                preds.append(int(candidate))

            # log mỗi 50 cửa sổ
            if (wi + 1) % 50 == 0 or (wi + 1) == len(labels):
                valid = [(p == t)
                         for p, t in zip(preds, labels[:len(preds)]) if p != -1]
                acc = np.mean(valid) if valid else 0
                print(f"{wi+1}/{len(labels)}  temp_acc={acc:.3f}")

    finally:
        ser.close()

    preds_arr = np.array(preds)
    labels_arr = np.array(labels)

    valid_mask = preds_arr >= 0
    preds_valid = preds_arr[valid_mask]
    labels_valid = labels_arr[valid_mask]

    # FINAL RESULT
    acc = (preds_valid == labels_valid).mean()
    print("\n=== FINAL RESULT ===")
    print("Accuracy:", acc)

    cm = confusion_matrix(labels_valid, preds_valid, labels=[1, 2, 3, 4])
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        labels_valid, preds_valid,
        labels=[1, 2, 3, 4], zero_division=0))

    pd.DataFrame({
        "true": labels_arr,
        "pred": preds_arr
    }).to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
