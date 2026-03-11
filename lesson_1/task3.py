import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)

df = pd.read_csv("/content/sample_data/house_price_regression_dataset.csv")

# Проверим есть ли столбцы с пропущенными значениями
df.isna().any()

# Подготовка данных
X = df.drop("House_Price", axis=1)
y = df["House_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказание
y_pred = model.predict(X_test)

# MSE (Среднеквадратичная ошибка): чем ближе к 0, тем лучше
mse = root_mean_squared_error(y_test, y_pred)

# RMSE (Корень из MSE): ошибка в тех же единицах, что и целевая переменная
# (например, в рублях или метрах)
rmse = root_mean_squared_error(y_test, y_pred, squared=False)

# MAE (Средняя абсолютная ошибка): среднее отклонение
mae = mean_absolute_error(y_test, y_pred)

# R^2 (Коэффициент детерминации): точность от 0 до 1
r2 = r2_score(y_test, y_pred)

# Вывод результатов
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
