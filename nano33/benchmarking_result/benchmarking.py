#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCU windowed tester
- đọc file CSV gốc (customer,activity,createdAt,x,y,z)
- tạo sliding windows theo create_training_data()
- gửi từng cửa sổ xuống MCU (gửi từng dòng x,y,z)
- nhận pred từ MCU (mỗi cửa sổ nhận 1 pred)
- so sánh với label cửa sổ, tính accuracy và confusion matrix
"""

import time
import serial
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ----------------
SERIAL_PORT = "COM5"
BAUDRATE = 115200
CSV_FILE = "test_div10.csv"
OUT_CSV = "mcu_test_result.csv"
WINDOW_SIZE_FOR_PC = 10
STEP_SIZE = 5
SERIAL_TIMEOUT = 2.0

LINE_DELAY = 0.001


def create_training_data(data, window_size=20, step_size=10):
    """
    Chia dữ liệu cảm biến thành các cửa sổ trượt (sliding windows)
    Trả về: x_list (list of arrays), y_list, z_list, train_labels
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
        train_labels.append(int(label))

    return x_list, y_list, z_list, train_labels

# ------------- Main ----------------


def main():
    df = pd.read_csv(CSV_FILE)
    for col in ['activity', 'x', 'y', 'z']:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {CSV_FILE}")

    x_windows, y_windows, z_windows, labels = create_training_data(
        df, window_size=WINDOW_SIZE_FOR_PC, step_size=STEP_SIZE
    )

    print(
        f"Total windows prepared: {len(labels)} (window_size={WINDOW_SIZE_FOR_PC}, step={STEP_SIZE})")
    if len(labels) == 0:
        raise SystemExit(
            "No valid windows found. Check window_size/step_size or labels in CSV.")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2.0)
    ser.flushInput()
    ser.flushOutput()
    print(f"Connected to {SERIAL_PORT} @ {BAUDRATE} baud")

    preds = []

    try:
        for wi in range(len(labels)):
            xw = x_windows[wi]
            yw = y_windows[wi]
            zw = z_windows[wi]

            if not (len(xw) == WINDOW_SIZE_FOR_PC and len(yw) == WINDOW_SIZE_FOR_PC and len(zw) == WINDOW_SIZE_FOR_PC):
                print(f"Skip window {wi}: invalid size")
                preds.append(-1)
                continue

            for j in range(WINDOW_SIZE_FOR_PC):
                line = f"{xw[j]},{yw[j]},{zw[j]}\n"
                ser.write(line.encode('utf-8'))
                time.sleep(LINE_DELAY)

            # Sau khi gửi đủ WINDOW_SIZE dòng, MCU sẽ in ra 1 dòng pred (số nguyên)
            # Đọc 1 dòng trả về
            try:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception as e:
                print(f"[ERROR] Serial read failed at window {wi}: {e}")
                raw = ""

            if raw == "":
                print(f"[WARN] Empty response for window {wi}; storing -1")
                preds.append(-1)
            else:
                token = None
                for part in raw.split():
                    if part.lstrip('-').isdigit():
                        token = part
                        break
                if token is None:
                    print(
                        f"[WARN] Non-integer MCU response for window {wi}: '{raw}'")
                    preds.append(-1)
                else:
                    preds.append(int(token))

            if (wi + 1) % 50 == 0 or (wi + 1) == len(labels):
                acc_sofar = np.mean([1 if p == t else 0 for p, t in zip(
                    preds, labels[:len(preds)]) if p != -1])
                print(
                    f"Window {wi+1}/{len(labels)} — temp accuracy (ignoring -1) = {acc_sofar:.4f}")

    finally:
        ser.close()

    preds_arr = np.array(preds)
    labels_arr = np.array(labels)

    valid_idx = preds_arr >= 0
    if valid_idx.sum() == 0:
        print("No valid predictions received.")
        return

    preds_valid = preds_arr[valid_idx]
    labels_valid = labels_arr[valid_idx]

    overall_acc = (preds_valid == labels_valid).mean()
    print("\n=== Final results ===")
    print(f"Total windows: {len(labels)}")
    print(f"Valid predictions: {valid_idx.sum()}")
    print(f"Accuracy: {overall_acc:.4f}")

    # confusion + report
    cm = confusion_matrix(labels_valid, preds_valid, labels=[1, 2, 3, 4])
    print("\nConfusion matrix (rows=true, cols=pred) labels [1,2,3,4]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(labels_valid, preds_valid,
          labels=[1, 2, 3, 4], zero_division=0))

    # 5) lưu kết quả chi tiết
    out_df = pd.DataFrame({
        "true": labels,
        "pred": preds
    })
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved detail to {OUT_CSV}")


if __name__ == "__main__":
    main()
