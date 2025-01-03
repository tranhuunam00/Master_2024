import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import joblib
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def create_training_data_NN(data):
    print("STARTING create_training_data_NN")

    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = {'x', 'y', 'z', 'activity'}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Dữ liệu đầu vào thiếu một trong các cột: {required_columns}")

    # Chuẩn hóa dữ liệu theo công thức: (x + 12) / 24
    total_list_NN = data[['x', 'y', 'z']].values.astype(np.float32)
    total_list_NN = (total_list_NN + 12) / 24  # Áp dụng chuẩn hóa

    # Chuyển đổi nhãn `activity` thành số nguyên liên tục từ 0 đến num_classes - 1
    unique_labels = np.unique(data['activity'])
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    train_labels_NN = np.array([label_mapping[label]
                               for label in data['activity']], dtype=np.int32)

    print(f"Nhãn sau khi chuyển đổi: {label_mapping}")

    return total_list_NN, train_labels_NN


def NeuralNetworkModel(train, test, labelTrain, labelTest):
    # Chuyển dữ liệu thành numpy array
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    # Định nghĩa số lớp output dựa trên số nhãn duy nhất
    num_classes = len(np.unique(labelTrain))

    model = keras.Sequential([
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        # Số lớp output = số nhãn
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train, labelTrain, epochs=30, verbose=1, batch_size=16)

    # Đánh giá mô hình trên tập test
    loss, accuracy = model.evaluate(test, labelTest)
    print("Accuracy:", accuracy)

    # Dự đoán trên tập test
    y_pred = np.argmax(model.predict(test), axis=1)

    # Tính toán ma trận nhầm lẫn
    cm = confusion_matrix(labelTest, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # In báo cáo phân loại
    print("\nClassification Report:\n", classification_report(labelTest, y_pred))

    return model


os.chdir('../positionSleep')
np.bool = np.bool_
np.int = np.int_


data = pd.read_csv('datas/train.csv')

data_test = pd.read_csv('datas/test.csv')

data['createdAt'] = data['createdAt'].str[:-38]
data_test['createdAt'] = data_test['createdAt'].str[:-38]


total_list_NN, train_labels_NN = create_training_data_NN(data=data)


print("total_list_NN")
print(total_list_NN)
print(train_labels_NN)

total_list_NN_test, train_labels_NN_test = create_training_data_NN(
    data=data_test)
trainNN = total_list_NN
testNN = total_list_NN_test
labelTrainNN = train_labels_NN
labelTestNN = train_labels_NN_test

print(len(trainNN))
print(len(labelTrainNN))


nnM = NeuralNetworkModel(train=trainNN, test=testNN,
                         labelTrain=labelTrainNN, labelTest=labelTestNN)

joblib.dump(nnM, 'namth.dat')
converter = tf.lite.TFLiteConverter.from_keras_model(nnM)
tflite_model = converter.convert()

# Save the model to disk
open("position.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("position.tflite")
print("Model is %d bytes" % basic_model_size)
# echo "const unsigned char model[] = {" > ./position_model.h
# cat position.tflite | xxd -i      >> ./position_model.h
# echo "};"                              >> ./position_model.h
