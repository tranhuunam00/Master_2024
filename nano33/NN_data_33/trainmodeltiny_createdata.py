import numpy as np
from scipy.stats import mode
import os


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


def get_keras_model_size(model, name="NeuralNetwork"):
    """Tính kích thước mô hình Keras (KB) và số tham số"""
    model_path = f"{name}_model.h5"
    model.save(model_path)  # Lưu mô hình Keras

    # Tính kích thước file (KB)
    model_kb = os.path.getsize(model_path) / 1024

    # Lấy số lượng tham số huấn luyện
    n_params = model.count_params()

    print(f"📦 {name}: Model Size = {model_kb:.2f} KB")
    print(f"🔢 Tổng số tham số huấn luyện: {n_params:,}")
    print("-" * 70)

    return model_kb, n_params

# Gọi hàm sau khi huấn luyện xong
