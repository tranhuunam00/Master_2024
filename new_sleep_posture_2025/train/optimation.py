from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np


def optimize_RF(train, labelTrain):
    # Các tham số cần dò
    param_grid = {
        # thêm 150, 200 → tăng độ mượt của rừng
        'n_estimators': [10, 20, 30],
        'max_depth': [10, 20, 30, None],       # thêm None để test full-depth
        # thêm 10 để test overfitting vs generalization
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],         # thêm 4 để tăng regularization
        'max_features': ['sqrt', 'log2'],  # thêm None cho auto features
        # thêm False để test full sampling
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(train, labelTrain)
    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)
    return grid_search.best_estimator_


def optimize_LR(train, test, labelTrain, labelTest):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    print("🔧 Starting Logistic Regression optimization...")

    # 1️⃣ Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    start = time.time()
    scaler.fit(train)

    end = time.time()
    print(f"⏱ Training time (scaler fit): {end - start:.4f}s")

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    # 2️⃣ Lưới tham số cần dò
    param_grid = {
        'C': [0.01, 0.1, 1, 5, 10, 50],              # độ mạnh regularization
        'solver': ['lbfgs', 'newton-cg', 'saga'],
        'penalty': ['l2'],
        'max_iter': [10, 20, 50]           # vừa đủ hội tụ
    }

    lr = LogisticRegression(random_state=21, multi_class='ovr')

    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # 3️⃣ Tối ưu hóa tham số
    grid_search.fit(train_scaled, labelTrain)
    best_lr = grid_search.best_estimator_

    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)

    # 4️⃣ Đánh giá mô hình trên tập test
    y_pred = best_lr.predict(test_scaled)
    acc = accuracy_score(labelTest, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print("\n------------- Classification Report (Optimized LR) -------------\n")
    print(classification_report(labelTest, y_pred))

    # 5️⃣ Ma trận nhầm lẫn
    cm = confusion_matrix(labelTest, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Optimized Logistic Regression")
    plt.show()

    # 6️⃣ Trả mô hình tối ưu nhất
    return best_lr, scaler


def optimize_SVM(train, test, labelTrain, labelTest):
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

    print("🔧 Starting Support Vector Machine optimization...")

    # 1️⃣ Chuẩn hoá dữ liệu
    scaler = StandardScaler()
    start = time.time()
    scaler.fit(train)
    end = time.time()
    print(f"⏱ Training time (scaler fit): {end - start:.4f}s")

    X_train_scaled = scaler.transform(train)
    X_test_scaled = scaler.transform(test)

    # 2️⃣ Định nghĩa lưới tham số cần dò
    param_grid = {
        'C': [0.1, 1, 2, 5, 10, 20],
        'kernel': ['linear', 'sigmoid', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 5],
        'decision_function_shape': ['ovo', 'ovr']
    }

    svm = SVC(random_state=42)

    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # 3️⃣ Huấn luyện và tối ưu hoá tham số
    grid_search.fit(X_train_scaled, labelTrain)
    best_svm = grid_search.best_estimator_

    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)

    # 4️⃣ Đánh giá trên tập test
    y_pred = best_svm.predict(X_test_scaled)
    acc = accuracy_score(labelTest, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print("\n------------- Classification Report (Optimized SVM) -------------\n")
    print(classification_report(labelTest, y_pred))

    # 5️⃣ Ma trận nhầm lẫn
    cm = confusion_matrix(labelTest, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Optimized SVM")
    plt.show()

    # 6️⃣ Trả về mô hình tối ưu và scaler
    return best_svm, scaler


def optimize_GB(train, test, labelTrain, labelTest):
    import time
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

    print("🔧 Starting Gradient Boosting optimization...")

    # 1️⃣ Chuẩn hoá dữ liệu
    scaler = StandardScaler()
    start = time.time()
    scaler.fit(train)
    end = time.time()
    print(f"⏱ Training time (scaler fit): {end - start:.4f}s")

    X_train_scaled = scaler.transform(train)
    X_test_scaled = scaler.transform(test)

    # 2️⃣ Định nghĩa lưới tham số để tối ưu
    param_grid = {
        'n_estimators': [10, 20, 50,],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'max_features': ['sqrt', 'log2', None],
        'subsample': [0.8, 1.0]
    }

    gb = GradientBoostingClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # 3️⃣ Huấn luyện và tìm tham số tốt nhất
    grid_search.fit(X_train_scaled, labelTrain)
    best_gb = grid_search.best_estimator_

    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)

    # 4️⃣ Đánh giá trên tập test
    y_pred = best_gb.predict(X_test_scaled)
    acc = accuracy_score(labelTest, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print("\n------------- Classification Report (Optimized GB) -------------\n")
    print(classification_report(labelTest, y_pred))

    # 5️⃣ Ma trận nhầm lẫn
    cm = confusion_matrix(labelTest, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Optimized Gradient Boosting")
    plt.show()

    # 6️⃣ Trả về mô hình tốt nhất và scaler
    return best_gb, scaler


def optimize_NN_raw(train, test, labelTrain, labelTest):
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    from scikeras.wrappers import KerasClassifier
    from tensorflow import keras
    from tensorflow.keras import layers

    print("🔧 Starting Neural Network (raw data) optimization...")

    # 1️⃣ Tiền xử lý dữ liệu
    train = np.array(train, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    labelTrain = np.array(labelTrain, dtype=np.int32)
    labelTest = np.array(labelTest, dtype=np.int32)

    # Dịch nhãn nếu cần
    if labelTrain.min() == 1:
        labelTrain -= 1
        labelTest -= 1

    num_classes = int(max(labelTrain.max(), labelTest.max()) + 1)
    print(f"🧩 num_classes = {num_classes}")

    # 2️⃣ Định nghĩa hàm xây dựng model (có thể thay đổi kiến trúc)
    def build_model(neurons_1=16, neurons_2=8, learning_rate=0.005):
        model = keras.Sequential([
            layers.Dense(neurons_1, activation='relu',
                         input_shape=(train.shape[1],)),
            layers.Dense(neurons_2, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # 3️⃣ Tạo wrapper cho GridSearchCV
    nn_model = KerasClassifier(model=build_model, verbose=0)

    # 4️⃣ Lưới tham số cần dò
    param_grid = {
        "model__neurons_1": [8, 16],
        "model__neurons_2": [4, 8,],
        "model__learning_rate": [0.01, 0.1],
        "batch_size": [16, 32],
        "epochs": [20, 30]
    }

    # Tổng cộng: 3×3×4×2×3 = 216 tổ hợp
    # cv=3 → 648 fits (khoảng 2-3 phút chạy)
    grid_search = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # 5️⃣ Huấn luyện & dò tham số
    start = time.time()
    grid_search.fit(train, labelTrain)
    end = time.time()

    print(f"⏱ Optimization time: {end - start:.2f}s")
    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)

    # 6️⃣ Đánh giá trên tập test
    best_nn = grid_search.best_estimator_
    y_pred = best_nn.predict(test)
    acc = accuracy_score(labelTest, y_pred)

    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print("\n------------- Classification Report (Optimized NN Raw) -------------\n")
    print(classification_report(labelTest, y_pred,
          target_names=["Back", "Right", "Left", "Stomach"]))

    # 7️⃣ Ma trận nhầm lẫn
    cm = confusion_matrix(labelTest, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                  "Back", "Right", "Left", "Stomach"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix - Optimized NN (Raw Data)")
    plt.show()

    return best_nn


def get_model_size_kb(model, scaler, name):
    """Lưu model & (nếu có) scaler, tính dung lượng và số tham số"""
    model_path = f"{name}_model.pkl"
    joblib.dump(model, model_path)
    model_kb = os.path.getsize(model_path) / 1024

    total_kb = model_kb
    scaler_kb = 0

    # 🔹 Lưu scaler nếu có
    if scaler is not None:
        scaler_path = f"{name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        scaler_kb = os.path.getsize(scaler_path) / 1024
        total_kb += scaler_kb

    # 🔹 In kích thước
    print(f"📦 {name}: Model = {model_kb:.2f} KB | Scaler = {scaler_kb:.2f} KB | Total = {total_kb:.2f} KB")

    # 🔹 Nếu là mô hình tuyến tính (LR, SVM)
    if hasattr(model, "coef_"):
        n_params = np.prod(model.coef_.shape) + len(model.intercept_)
        print(f"🔢  → Số tham số huấn luyện: {n_params}")

    # 🔹 Nếu là mô hình cây (RF, GB)
    elif hasattr(model, "estimators_"):
        try:
            n_nodes = 0
            for est in model.estimators_:
                # GradientBoosting có thể là mảng 2D các cây con
                if isinstance(est, (list, np.ndarray)):
                    for sub_est in est:
                        if hasattr(sub_est, "tree_"):
                            n_nodes += sub_est.tree_.node_count
                else:
                    if hasattr(est, "tree_"):
                        n_nodes += est.tree_.node_count
            print(f"🌲  → Tổng số nút trong mô hình cây: {n_nodes}")
        except Exception as e:
            print(f"⚠️  Không thể đếm số nút (lý do: {e})")

    print("-" * 70)
    return total_kb
