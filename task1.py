import numpy as np
from sklearn.model_selection import train_test_split

# Создаем массив размера (40, 3)
array = np.random.randint(10, 101, size=(40, 3))
array[:, 2] = np.random.randint(2, size=40)

# Разделяем на признаки и таргет
data = array[:, :2]
target = array[:, 2]

# Разбиваем на обучающую и тестовую выборки
numbers_train, numbers_test, binary_train, binary_test = train_test_split(
    data, target, test_size=0.3, random_state=10
)

print(f"Размер numbers_train: {numbers_train.shape}")
print(f"Размер numbers_test: {numbers_test.shape}")