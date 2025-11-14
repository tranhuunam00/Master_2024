import numpy as np
import pandas as pd
import os
# import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from trainmodeltiny_createdata import create_training_data_NN_like_micro, get_keras_model_size
from sklearn.metrics import accuracy_score, confusion_matrix


def NeuralNetworkModel(train, test, labelTrain, labelTest):
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    if labelTrain.min() == 1:
        labelTrain -= 1
        labelTest -= 1

    num_classes = int(max(labelTrain.max(), labelTest.max()) + 1)

    model = keras.Sequential([
        keras.Input(shape=(train.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt,
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


data = pd.read_csv('./input_nano_33/train_div10.csv')
data_test = pd.read_csv('./input_nano_33/test_div10.csv')


total_list_NN, train_labels_NN = create_training_data_NN_like_micro(data=data)
total_list_NN_test, train_labels_NN_test = create_training_data_NN_like_micro(
    data=data_test)

nnM = NeuralNetworkModel(train=total_list_NN, test=total_list_NN_test,
                         labelTrain=train_labels_NN, labelTest=train_labels_NN_test)


get_keras_model_size(nnM, "NeuralNetwork")

# joblib.dump(nnM, 'namth.dat')
converter = tf.lite.TFLiteConverter.from_keras_model(nnM)
tflite_model = converter.convert()

# Save the model to disk
open("position.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("position.tflite")
print("Model is %d bytes" % basic_model_size)
# echo "const unsigned char model[] = {" > ./position_model.h
# cat position.tflite | xxd -i      >> ./position_model.h
# echo "};"                              >> ./position_model.h
