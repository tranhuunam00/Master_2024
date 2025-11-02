from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def optimize_RF(train, labelTrain):
    # Các tham số cần dò
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,               # 5-fold cross-validation
        n_jobs=1,          # chạy song song
        verbose=2,
        scoring='accuracy'  # hoặc 'f1_macro' nếu dữ liệu mất cân bằng
    )

    grid_search.fit(train, labelTrain)
    print("✅ Best Params:", grid_search.best_params_)
    print("✅ Best CV Score:", grid_search.best_score_)
    return grid_search.best_estimator_
