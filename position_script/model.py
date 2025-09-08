from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from util import matrixConf
from scipy.stats import skew, kurtosis
import math
import array as arr
import warnings
from scipy.signal import find_peaks
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import svm, datasets
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
np.bool = np.bool_
np.int = np.int_


def LRmodel(train, test, labelTrain, labelTest):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    # standardization
    scaler = StandardScaler()
    start = time.time()

    scaler.fit(train)
    end = time.time()
    print("training time LRmodel complexity :", end-start, "s")

    trainScaler = scaler.transform(train)
    testScaler = scaler.transform(test)
    # logistic regression model
    lr = LogisticRegression(random_state=21,  max_iter=100,  multi_class="ovr")
    lr.fit(trainScaler, labelTrain)
    y_pred = lr.predict(testScaler)
    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification LogisticRegression Report-------------\n")
    print(classification_report(labelTest, y_pred))
    matrixConf(labelTest, y_pred)
    return lr


def RFmodel(train, test, labelTrain, labelTest):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    # Thay đổi SVC thành RandomForestClassifier
    rfc = RandomForestClassifier(
        n_estimators=50, max_depth=None, random_state=42, max_features="log2")

    start = time.time()
    rfc.fit(train, labelTrain)
    end = time.time()

    print("training time complexity :", end-start, "s")

    y_pred = rfc.predict(test)
    confusion_matrix(labelTest, y_pred)

    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification RandomForestClassifier Report-------------\n")
    print(classification_report(labelTest, y_pred))
    matrixConf(labelTest, y_pred)

    return rfc


def SVMmodel(train, test, labelTrain, labelTest):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import time
    # Thay thế RandomForestClassifier bằng SVM
    svm_model = SVC(kernel='sigmoid', C=2.0, random_state=42,
                    decision_function_shape="ovo")

    start = time.time()
    svm_model.fit(train, labelTrain)
    end = time.time()

    print("training time complexity :", end-start, "s")

    y_pred = svm_model.predict(test)
    confusion_matrix(labelTest, y_pred)

    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification SVM Report-------------\n")
    print(classification_report(labelTest, y_pred))
    matrixConf(labelTest, y_pred)

    return svm_model


def GradientBoostingModel(train, test, labelTrain, labelTest):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import time

    # Thay thế SVM bằng Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.01, random_state=42, max_depth=4, max_features="log2")

    start = time.time()
    gb_model.fit(train, labelTrain)
    end = time.time()

    print("Training time complexity:", end - start, "s")

    y_pred = gb_model.predict(test)
    confusion_matrix(labelTest, y_pred)

    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification Gradient Boosting Report-------------\n")
    print(classification_report(labelTest, y_pred))
    matrixConf(labelTest, y_pred)

    return gb_model


def NeuralNetworkModel(train, test, labelTrain, labelTest):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import time
    # Thay thế Gradient Boosting bằng Neural Network
    nn_model = MLPClassifier(
        hidden_layer_sizes=(8, 4),  # Số lượng và kích thước các lớp ẩn
        activation='relu',          # Hàm kích hoạt của các nút
        solver='adam',              # Thuật toán tối ưu hóa
        # Số lượng epoch (vòng lặp qua toàn bộ tập dữ liệu)
        max_iter=100,
        random_state=42,
        learning_rate_init=0.01
    )

    start = time.time()
    nn_model.fit(train, labelTrain)
    end = time.time()

    print("Training time complexity:", end - start, "s")

    y_pred = nn_model.predict(test)
    confusion_matrix(labelTest, y_pred)

    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification Neural Network Report-------------\n")
    print(classification_report(labelTest, y_pred))
    matrixConf(labelTest, y_pred)

    return nn_model


def NeuralNetworkModel2(train, test, labelTrain, labelTest):
    # Convert numpy array
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    # Dịch nhãn về 0-based
    labelTrain = labelTrain - 1
    labelTest = labelTest - 1

    num_classes = int(max(labelTrain.max(), labelTest.max()) + 1)
    print("Unique labels train after shift:", np.unique(labelTrain))
    print("Unique labels test after shift:", np.unique(labelTest))
    print("num_classes set to:", num_classes)

    # Define model
    model = keras.Sequential([
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    start = time.time()
    model.fit(train, labelTrain, epochs=30, verbose=1, batch_size=16)
    end = time.time()
    print("Training time complexity:", end - start, "s")

    # Predict
    y_pred_prob = model.predict(test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Evaluation
    print("Accuracy:", accuracy_score(labelTest, y_pred))
    print("\n -------------Classification Neural Network Report-------------\n")
    print(classification_report(labelTest, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(labelTest, y_pred))

    return model
