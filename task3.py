import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

# Датасет 6: таргет - уровень премии (низкий/средний/высокий)
df6 = pd.DataFrame({
    'Completion_Pct': [10, 25, 45, 50, 75, 85, 95, 100],
    'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': ['Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High']
})

# Масштабирование признака Completion_Pct с помощью MinMaxScaler
# Обоснование: MinMaxScaler нормализует значения в диапазон [0, 1], 
# что хорошо подходит для процентов выполнения (Completion_Pct)
scaler = MinMaxScaler()
df6[['Completion_Pct']] = scaler.fit_transform(df6[['Completion_Pct']])

# Кодирование таргета с помощью LabelEncoder
# Обоснование: Уровни премии (Low, Medium, High) имеют естественный порядок,
# но LabelEncoder назначит им метки 0, 1, 2 соответственно, сохранив этот порядок
le = LabelEncoder()
df6['Target'] = le.fit_transform(df6['Target'])

print("Датасет 6:")
print(df6)
print()

# Датасет 7: таргет - одобрение кредита (да/нет)
df7 = pd.DataFrame({
    'Income_K': [30, 35, 40, 45, 50, 42, 38, 1000],
    'Credit_Score': [600, 620, 640, 610, 650, 630, 615, 800],
    'Target': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

# Масштабирование признака Income_K с помощью StandardScaler
# Обоснование: StandardScaler стандартизирует данные (среднее = 0, std = 1),
# что особенно важно при наличии выброса (1000) в данных о доходе
scaler = StandardScaler()
df7[['Income_K']] = scaler.fit_transform(df7[['Income_K']])

# Кодирование таргета с помощью LabelEncoder
# Обоснование: Бинарная классификация (Yes/No), LabelEncoder преобразует в 1/0
le = LabelEncoder()
df7['Target'] = le.fit_transform(df7['Target'])

print("Датасет 7:")
print(df7)