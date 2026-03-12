import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import recall_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

# Загрузка и подготовка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = 1 - pd.Series(data.target)  # 1 = злокачественная

X_train, X_test, y_train, y_test = train_test_split(
                                                X, y, test_size=0.2,
                                                random_state=42,
                                                stratify=y
                                            )


# Создание дополнительных признаков
def add_features(X):
    X = X.copy()
    X['area_ratio'] = X['mean area'] / (X['mean radius']**2 + 1e-6)
    X['concavity_worst'] = X['worst concave points'] * X['worst area']
    X['log_area'] = np.log1p(X['worst area'])
    return X


X_train = add_features(X_train)
X_test = add_features(X_test)

# Оптимизированная модель
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(f_classif, k=20)),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear',
                               random_state=42))
])

# Поиск лучших параметров по recall
param_grid = {
    'select__k': [15, 20, 25],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid.fit(X_train, y_train)

# Настройка порога
y_proba = grid.predict_proba(X_train)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba)
optimal_threshold = thresholds[np.argmax(recalls[:-1] * (precisions[:-1] >= 0.7))]

# Финальные предсказания
y_pred = (grid.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

# Результаты
print(f"Лучшие параметры: {grid.best_params_}")
print(f"Оптимальный порог: {optimal_threshold:.3f}")
print(f"\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Доброкач.', 'Злокач.']))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Recall (злокачественные): {recall_score(y_test, y_pred):.4f}")
