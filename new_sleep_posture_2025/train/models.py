"""
optimized_models.py
Author: Ngoc Thai Tran et al.
Date: 2025
Description:
    Pre-optimized ML models for sleep posture detection
    (used after grid search fine-tuning)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================================
# Utility
# ==========================================================


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

# ==========================================================
# 1️⃣ Random Forest
# ✅ Best Params: {'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt',
#                 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 20}
# ==========================================================


def train_RF(train, test, labelTrain, labelTest):
    print("🌲 Training Random Forest (optimized)...")
    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        random_state=42
    )
    rf.fit(train, labelTrain)
    y_pred = rf.predict(test)
    acc = accuracy_score(labelTest, y_pred)
    print(f"🎯 Test Accuracy: {acc:.4f}")
    print(classification_report(labelTest, y_pred))
    plot_confusion_matrix(labelTest, y_pred, "Random Forest (Optimized)")
    return rf


# ==========================================================
# 2️⃣ Logistic Regression
# ✅ Best Params: {'C': 5, 'max_iter': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
# ==========================================================

def train_LR(train, test, labelTrain, labelTest):
    print("📈 Training Logistic Regression (optimized)...")
    scaler = StandardScaler()
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)

    lr = LogisticRegression(
        C=5,
        max_iter=10,
        penalty='l2',
        solver='newton-cg',
        random_state=21,
        multi_class='ovr'
    )

    lr.fit(X_train, labelTrain)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(labelTest, y_pred)
    print(f"🎯 Test Accuracy: {acc:.4f}")
    print(classification_report(labelTest, y_pred))
    plot_confusion_matrix(labelTest, y_pred, "Logistic Regression (Optimized)")
    return lr, scaler


# ==========================================================
# 3️⃣ SVM
# ✅ Best Params: {'C': 1, 'decision_function_shape': 'ovo',
#                 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
# ==========================================================

def train_SVM(train, test, labelTrain, labelTest):
    print("💡 Training SVM (optimized)...")
    scaler = StandardScaler()
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)

    svm = SVC(
        C=1,
        kernel='linear',
        gamma='scale',
        degree=2,
        decision_function_shape='ovo',
        random_state=42
    )

    svm.fit(X_train, labelTrain)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(labelTest, y_pred)
    print(f"🎯 Test Accuracy: {acc:.4f}")
    print(classification_report(labelTest, y_pred))
    plot_confusion_matrix(labelTest, y_pred, "SVM (Optimized)")
    return svm, scaler


# ==========================================================
# 4️⃣ Gradient Boosting
# ✅ Best Params: {'learning_rate': 0.1, 'max_depth': 4,
#                 'max_features': 'sqrt', 'n_estimators': 10, 'subsample': 0.8}
# ==========================================================

def train_GB(train, test, labelTrain, labelTest):
    print("🔥 Training Gradient Boosting (optimized)...")
    scaler = StandardScaler()
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)

    gb = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=4,
        max_features='sqrt',
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )

    gb.fit(X_train, labelTrain)
    y_pred = gb.predict(X_test)
    acc = accuracy_score(labelTest, y_pred)
    print(f"🎯 Test Accuracy: {acc:.4f}")
    print(classification_report(labelTest, y_pred))
    plot_confusion_matrix(labelTest, y_pred, "Gradient Boosting (Optimized)")
    return gb, scaler


# ==========================================================
# 5️⃣ Neural Network (Keras)
# ✅ Best Params: {'batch_size': 32, 'epochs': 20,
#                 'model__learning_rate': 0.01, 'model__neurons_1': 8, 'model__neurons_2': 8}
# ==========================================================

def train_NN_raw(train, test, labelTrain, labelTest):
    print("🧠 Training Neural Network (optimized)...")
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    if labelTrain.min() == 1:
        labelTrain -= 1
        labelTest -= 1

    num_classes = int(max(labelTrain.max(), labelTest.max()) + 1)
    print(f"🧩 Classes: {num_classes}")

    model = keras.Sequential([
        keras.Input(shape=(train.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    model.fit(train, labelTrain, batch_size=32, epochs=20, verbose=1)
    print(f"⏱ Training time: {time.time() - start:.2f}s")

    loss, acc = model.evaluate(test, labelTest, verbose=0)
    print(f"🎯 Test Accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(test), axis=1)
    print(classification_report(labelTest, y_pred,
          target_names=["Back", "Right", "Left", "Stomach"]))
    plot_confusion_matrix(labelTest, y_pred, "Neural Network (Optimized)")
    return model
