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

# Список значений C
Cs = [0.001, 0.01, 0.1, 1, 10]

print("C\tНулевых весов\tТочность")
print("-" * 35)

for c in Cs:
    # Создание и обучение модели с L1
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=c, multi_class="ovr"
    )
    model.fit(X_train, y_train)

    # Подсчет нулевых весов
    n_zero = np.sum(model.coef_ == 0)

    # Оценка точности
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{c}\t{n_zero}\t\t{acc:.3f}")
