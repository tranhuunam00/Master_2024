import numpy as np
from scipy.stats import mode


def create_training_data_NN(data):
    print("STARTING create_training_data_NN")
    WINDOW_LENGTH = 10  # Kích thước cửa sổ
    OVERLAP = 5

    required_columns = {'x', 'y', 'z', 'activity'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Dữ liệu đầu vào thiếu các cột: {missing}")

    features = ['x', 'y', 'z']
    total_list_NN = data[features].values.astype(np.float32)
    # total_list_NN = (total_list_NN + 1.2)

    unique_labels = np.unique(data['activity'])
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    data['activity_num'] = data['activity'].map(label_mapping)

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
            window_label = mode(window_labels)[0]
            y.append(window_label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y
