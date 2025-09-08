import numpy as np
from scipy.stats import mode


def create_training_data_NN(data):
    print("STARTING create_training_data_NN")
    WINDOW_LENGTH = 20  # Kích thước cửa sổ
    OVERLAP = 10        # Số mẫu trùng lặp (50% của WINDOW_LENGTH)

    # 1. Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = {'x', 'y', 'z', 'activity'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Dữ liệu đầu vào thiếu các cột: {missing}")

    # 2. Chuẩn hóa dữ liệu theo công thức: (x + 12) / 24
    # lý do 12 vì giới hạn của acc value = 11
    features = ['x', 'y', 'z']
    total_list_NN = data[features].values.astype(np.float32)
    total_list_NN = (total_list_NN + 1.2)

    # 3. Chuyển đổi nhãn `activity` thành số nguyên liên tục từ 0 đến num_classes - 1
    unique_labels = np.unique(data['activity'])
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    data['activity_num'] = data['activity'].map(label_mapping)

    # 4. Tạo các cửa sổ dữ liệu với độ trùng lặp 50%
    X = []
    y = []
    num_samples = len(data)
    for start in range(0, num_samples - WINDOW_LENGTH + 1, OVERLAP):
        end = start + WINDOW_LENGTH
        window = total_list_NN[start:end]
        window_labels = data['activity_num'].iloc[start:end]

        if len(window) == WINDOW_LENGTH:
            flattened_window = window.flatten()
            X.append(flattened_window)
            # Gán nhãn cho cửa sổ bằng cách lấy nhãn xuất hiện nhiều nhất trong cửa sổ
            window_label = mode(window_labels)[0]
            y.append(window_label)

    if (len(X) == 20):
        X = np.array(X)
        y = np.array(y, dtype=np.int32)

    return X, y
