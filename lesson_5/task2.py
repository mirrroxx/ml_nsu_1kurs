import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

# Загрузка и подготовка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = 1 - pd.Series(data.target)  # 1 = злокачественная

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Создание дополнительных признаков с проверками
def add_features(X):
    X = X.copy()

    # Проверяем существование признаков перед созданием новых
    if "mean radius" in X.columns and "mean area" in X.columns:
        X["radius_area_ratio"] = X["mean area"] / (
            X["mean radius"] ** 2 + 1e-6
        )

    if "worst concave points" in X.columns and "worst area" in X.columns:
        X["concavity_risk"] = X["worst concave points"] * X["worst area"]

    if "worst area" in X.columns:
        X["log_worst_area"] = np.log1p(X["worst area"])

    if "mean perimeter" in X.columns and "mean radius" in X.columns:
        X["perimeter_radius_ratio"] = X["mean perimeter"] / (
            X["mean radius"] + 1e-6
        )

    return X


X_train = add_features(X_train)
X_test = add_features(X_test)

# Модель с оптимизацией по recall
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif)),
        (
            "clf",
            LogisticRegression(
                class_weight={0: 1, 1: 5},  # Усиливаем вес злокачественных
                solver="liblinear",
                random_state=42,
                max_iter=1000,
            ),
        ),
    ]
)

# Поиск лучших параметров
param_grid = {
    "select__k": [15, 20, 25, 30],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",  # Оптимизируем recall
    n_jobs=-1,
)
grid.fit(X_train, y_train)

# Оптимизация порога для минимизации FN
y_proba = grid.predict_proba(X_train)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba)

# Находим порог, дающий максимальный recall
# (даже если precision упадет, главное - не пропустить злокачественные)
best_recall = 0
best_threshold = 0.5

for recall, threshold in zip(recalls[:-1], thresholds):
    if recall > best_recall:
        best_recall = recall
        best_threshold = threshold

# Финальные предсказания с оптимизированным порогом
y_pred = (grid.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)

# Результаты
print("=" * 60)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ (ОПТИМИЗАЦИЯ ПО RECALL)")
print("=" * 60)
print(f"Лучшие параметры: {grid.best_params_}")
print(f"Оптимальный порог: {best_threshold:.3f}")
print(f"Recall на обучении: {best_recall:.4f}")
print("\n" + "=" * 60)
print("ТЕСТОВЫЕ ДАННЫЕ:")
print("=" * 60)
print(
    classification_report(
        y_test, y_pred, target_names=["Доброкач.", "Злокач."]
    )
)

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"              Предсказано")
print(f"              Добр  Злок")
print(f"Факт Добр:    {cm[0][0]:3d}  {cm[0][1]:3d}")
print(f"Факт Злок:    {cm[1][0]:3d}  {cm[1][1]:3d}")
print(f"\nFN (пропущенные злокачественные): {cm[1][0]}")
print(f"Recall для злокачественных: {recall_score(y_test, y_pred):.4f}")
