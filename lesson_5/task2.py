import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка и стандартизация данных
data = load_iris()
X, y = data.data, data.target
X_std = StandardScaler().fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.3, random_state=42
)

# Модель с L1
model_l1 = LogisticRegression(
    penalty="l1", solver="liblinear", C=0.01, multi_class="ovr"
)
model_l1.fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)
acc_l1 = accuracy_score(y_test, y_pred_l1)

# Модель с L2
model_l2 = LogisticRegression(
    penalty="l2", solver="lbfgs", C=0.01, multi_class="ovr"
)
model_l2.fit(X_train, y_train)
y_pred_l2 = model_l2.predict(X_test)
acc_l2 = accuracy_score(y_test, y_pred_l2)

# Вывод весов для первого класса
print("Веса признаков (первый класс):")
print(f"L1 weights: {model_l1.coef_[0]}")
print(f"L2 weights: {model_l2.coef_[0]}")

print(f"\nТочность L1: {acc_l1:.3f}")
print(f"Точность L2: {acc_l2:.3f}")

# Анализ
print("\nАнализ:")
if np.sum(model_l1.coef_[0] == 0) > 0:
    print("- В L1 есть нулевые веса (признаки отсечены)")
    print("- В L1 наблюдается резкий контраст между признаками")
    print(
        "- В L2 веса распределены более равномерно (все признаки учитываются)"
    )
