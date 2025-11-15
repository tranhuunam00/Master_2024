import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
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


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name):
    """Evaluate model on all sets and print metrics"""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_val = accuracy_score(y_val, y_val_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    print(f"\n{'='*60}")
    print(f" {name} Evaluation Results")
    print(f" Training Accuracy: {acc_train:.4f}")
    print(f" Validation Accuracy: {acc_val:.4f}")
    print(f" Test Accuracy: {acc_test:.4f}")

    print("\n--- Validation Report ---")
    print(classification_report(y_val, y_val_pred))
    print("\n--- Test Report ---")
    print(classification_report(y_test, y_test_pred))

    plot_confusion_matrix(y_val, y_val_pred, f"{name} - Validation")
    plot_confusion_matrix(y_test, y_test_pred, f"{name} - Test")

# ==========================================================
# 1️⃣ Random Forest
# ==========================================================


def train_RF(train, test, labelTrain, labelTest):
    print(" Training Random Forest (optimized)...")
    X_train, X_val, y_train, y_val = train_test_split(
        train, labelTrain, test_size=0.2, random_state=42, stratify=labelTrain
    )

    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        random_state=42
    )

    rf.fit(X_train, y_train)
    evaluate_model(rf, X_train, y_train, X_val, y_val,
                   test, labelTest, "Random Forest")
    return rf


# ==========================================================
# 2️⃣ Logistic Regression
# ==========================================================

def train_LR(train, test, labelTrain, labelTest):
    print(" Training Logistic Regression (optimized)...")

    scaler = StandardScaler()
    scaler.fit(train)
    X_train_scaled = scaler.transform(train)
    X_test_scaled = scaler.transform(test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, labelTrain, test_size=0.2, random_state=42, stratify=labelTrain
    )

    lr = LogisticRegression(
        C=5,
        max_iter=10,
        penalty='l2',
        solver='newton-cg',
        random_state=21,
        multi_class='ovr'
    )

    lr.fit(X_train, y_train)
    evaluate_model(lr, X_train, y_train, X_val, y_val,
                   X_test_scaled, labelTest, "Logistic Regression")
    return lr, scaler


# ==========================================================
# 3️⃣ SVM
# ==========================================================

def train_SVM(train, test, labelTrain, labelTest):
    print("💡 Training SVM (optimized)...")

    scaler = StandardScaler()
    scaler.fit(train)
    X_train_scaled = scaler.transform(train)
    X_test_scaled = scaler.transform(test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, labelTrain, test_size=0.2, random_state=42, stratify=labelTrain
    )

    svm = SVC(
        C=1,
        kernel='linear',
        gamma='scale',
        degree=2,
        decision_function_shape='ovo',
        random_state=42
    )

    svm.fit(X_train, y_train)
    evaluate_model(svm, X_train, y_train, X_val, y_val,
                   X_test_scaled, labelTest, "Support Vector Machine")
    return svm, scaler


# ==========================================================
# 4️⃣ Gradient Boosting
# ==========================================================

def train_GB(train, test, labelTrain, labelTest):
    print("🔥 Training Gradient Boosting (optimized)...")

    scaler = StandardScaler()
    scaler.fit(train)
    X_train_scaled = scaler.transform(train)
    X_test_scaled = scaler.transform(test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, labelTrain, test_size=0.2, random_state=42, stratify=labelTrain
    )

    gb = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=4,
        max_features='sqrt',
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )

    gb.fit(X_train, y_train)
    evaluate_model(gb, X_train, y_train, X_val, y_val,
                   X_test_scaled, labelTest, "Gradient Boosting")
    return gb, scaler


# ==========================================================
# 5️⃣ Neural Network (Keras)
# ==========================================================

def train_NN_raw(train, test, labelTrain, labelTest):
    print(" Training Neural Network (optimized)...")

    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    if labelTrain.min() == 1:
        labelTrain -= 1
        labelTest -= 1

    num_classes = int(max(labelTrain.max(), labelTest.max()) + 1)

    X_train, X_val, y_train, y_val = train_test_split(
        train, labelTrain, test_size=0.2, random_state=42, stratify=labelTrain
    )

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
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=20,
        verbose=1,
        validation_data=(X_val, y_val)
    )
    print(f"⏱ Training time: {time.time() - start:.2f}s")

    # Evaluate on test set
    loss_test, acc_test = model.evaluate(test, labelTest, verbose=0)
    loss_val, acc_val = model.evaluate(X_val, y_val, verbose=0)
    print(f" Validation Accuracy: {acc_val:.4f}")
    print(f" Test Accuracy: {acc_test:.4f}")

    y_pred_val = np.argmax(model.predict(X_val), axis=1)
    y_pred_test = np.argmax(model.predict(test), axis=1)

    print("\n--- Validation Report ---")
    print(classification_report(y_val, y_pred_val))
    print("\n--- Test Report ---")
    print(classification_report(labelTest, y_pred_test))

    plot_confusion_matrix(y_val, y_pred_val, "Neural Network - Validation")
    plot_confusion_matrix(labelTest, y_pred_test, "Neural Network - Test")

    return model, history
