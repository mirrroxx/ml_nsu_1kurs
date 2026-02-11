import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# 1. Датасет: Купит ли клиент товар
df1 = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50],
    'ID_System': [np.nan, 102, np.nan, 105, np.nan, 107],
    'Target': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes']
})

# Обработка NaN: ID_System - числовая переменная, используем среднее значение
# Обоснование: ID_System похож на идентификатор, но содержит пропуски. 
# Заполнение средним сохраняет общее распределение значений
imputer = SimpleImputer(strategy='mean')
df1[['ID_System']] = imputer.fit_transform(df1[['ID_System']])

# Кодирование таргета: Yes/No -> 1/0
# Обоснование: Бинарная классификация, 'No' -> 0 (отрицательный класс), 'Yes' -> 1 (положительный класс)
df1['Target'] = df1['Target'].map({'No': 0, 'Yes': 1})
print("Датасет 1:")
print(df1)
print()

# 2. Датасет: Уровень подписки
df2 = pd.DataFrame({
    'City': ['Moscow', 'Moscow', 'London', 'Moscow', np.nan, 'Moscow', 'London'],
    'Age': [20, 25, 30, 35, 40, 45, 50],
    'Target': ['Basic', 'Basic', 'Silver', 'Silver', 'Gold', 'Gold', 'Gold']
})

# Обработка NaN: City - категориальная переменная с пропуском, удаляем строку
# Обоснование: Всего 7 строк, один пропуск. Удаление минимально повлияет на данные
df2 = df2.dropna()

# Кодирование таргета: OrdinalEncoder для упорядоченных категорий
# Обоснование: Basic < Silver < Gold - явный порядок, требует ordinal encoding
encoder = OrdinalEncoder(categories=[['Basic', 'Silver', 'Gold']])
df2[['Target']] = encoder.fit_transform(df2[['Target']])
print("Датасет 2:")
print(df2)
print()

# 3. Датасет: Группа здоровья
df3 = pd.DataFrame({
    'Pulse': [70, 72, 75, np.nan, 68, 71, 73, 74],
    'Temp': [36.6, 36.7, 36.8, 36.6, 36.9, 36.6, 36.7, 36.8],
    'Target': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'C']
})

# Обработка NaN: Pulse - важный медицинский показатель, удаляем строку
# Обоснование: В медицинских данных лучше удалить строку с пропуском важного показателя
df3 = df3.dropna()

# Кодирование таргета: OrdinalEncoder для упорядоченных категорий
# Обоснование: A < B < C - группы здоровья имеют естественный порядок
encoder = OrdinalEncoder(categories=[['A', 'B', 'C']])
df3[['Target']] = encoder.fit_transform(df3[['Target']])
print("Датасет 3:")
print(df3)
print()

# 4. Датасет: Прошел проверку безопасности
df4 = pd.DataFrame({
    'Days_Since_Last_Incident': [10, 5, 20, np.nan, 15, 30],
    'Risk_Score': [0.1, 0.2, 0.1, 0.4, 0.2, 0.1],
    'Target': ['Safe', 'Safe', 'Warning', 'Safe', 'Safe', 'Warning']
})

# Обработка NaN: Days_Since_Last_Incident - временной показатель, удаляем строку
# Обоснование: В данных безопасности лучше иметь полные записи, строка с пропуском удаляется
df4 = df4.dropna()

# Кодирование таргета: LabelEncoder для НЕупорядоченных категорий
# Обоснование: 'Safe' и 'Warning' - разные категории без явного порядка
encoder = LabelEncoder()
df4['Target'] = encoder.fit_transform(df4['Target'])
print("Датасет 4:")
print(df4)
print()

# 5. Датасет: Кредитный рейтинг
df5 = pd.DataFrame({
    'Bonus_Points': [100, 500, np.nan, 200, np.nan, 800],
    'Salary_K': [50, 100, 40, 120, 30, 150],
    'Target': ['Low', 'High', 'Low', 'High', 'Low', 'High']
})

# Обработка NaN: Bonus_Points - числовая переменная, заполняем средним
# Обоснование: Bonus_Points можно заполнить средним значением для сохранения распределения
imputer = SimpleImputer(strategy='mean')
df5['Bonus_Points'] = imputer.fit_transform(df5[['Bonus_Points']])

# Кодирование таргета: OrdinalEncoder для упорядоченных категорий
# Обоснование: Low < High - кредитный рейтинг имеет явный порядок
encoder = OrdinalEncoder(categories=[['Low', 'High']])
df5[['Target']] = encoder.fit_transform(df5[['Target']])
print("Датасет 5:")
print(df5)