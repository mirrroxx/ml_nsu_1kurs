# 1. Импорт библиотек
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. Загрузка данных
df = pd.read_csv("/content/housing.csv")
print(f"Исходный размер данных: {df.shape}")

# 3. Очистка данных
df = df.dropna()  # удаляем пропуски
print(f"После удаления пропусков: {df.shape}")

# 4. Масштабирование целевой переменной
if df["median_house_value"].max() > 100:
    df["median_house_value"] /= 100000

# 5. Кодирование категориального признака
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# 6. Удаление выбросов и логарифмирование
df = df[df["median_house_value"] <= 5.0]
df["median_house_value"] = np.log(df["median_house_value"])

# 7. Удаление ненужных признаков и создание новых
df = df.drop(columns=["total_bedrooms", "latitude"], errors="ignore")
df["HouseAge_squared"] = df["housing_median_age"] ** 2
df["Population_squared"] = df["population"] ** 2
df["MedInc_squared"] = df["median_income"] ** 2

print(f"Финальный размер данных: {df.shape}")
print(df.head())

# 8. Подготовка признаков и целевой переменной
X = df.drop("median_house_value", axis=1).values
y = df["median_house_value"].values.reshape(-1, 1)

# 9. Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Train размер: {X_train.shape}, Test размер: {X_test.shape}")

# 10. Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Конвертация в тензоры PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 12. Создание модели
model = nn.Linear(X_train.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.MSELoss()

# 13. Обучение модели
epochs = 500
history = []

for epoch in range(epochs):
    # Прямой проход
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)

    # Обратный проход
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Сохранение истории
    history.append(loss.item())

    # Вывод каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 14. Визуализация результатов
plt.figure(figsize=(15, 5))

# График обучения
plt.subplot(1, 3, 1)
plt.plot(history)
plt.title("Процесс обучения (MSE)")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.grid(True)

# Анализ остатков
with torch.no_grad():
    test_preds = model(X_test_t).numpy()
    residuals = y_test - test_preds

plt.subplot(1, 3, 2)
plt.scatter(test_preds, residuals, alpha=0.3, color="teal")
plt.axhline(0, color="red", linestyle="--")
plt.title("Анализ остатков")
plt.xlabel("Предсказанная цена")
plt.ylabel("Ошибка")
plt.grid(True)

# Корреляционная матрица
plt.subplot(1, 3, 3)
corr_matrix = df.corr()
sns.heatmap(
    corr_matrix[["median_house_value"]].sort_values(
        by="median_house_value", ascending=False
    ),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    cbar=False,
)
plt.title("Корреляция с ценой")

plt.tight_layout()
plt.show()

# 15. Финальная оценка
with torch.no_grad():
    train_preds = model(X_train_t).numpy()
    test_preds = model(X_test_t).numpy()
    train_mse = criterion(torch.tensor(train_preds), y_train_t).item()
    test_mse = criterion(torch.tensor(test_preds), y_test_t).item()

print(f"\nФинальные результаты:")
print(f"MSE на train: {train_mse:.4f}")
print(f"MSE на test: {test_mse:.4f}")
print(f"Разница (переобучение): {abs(train_mse - test_mse):.4f}")
