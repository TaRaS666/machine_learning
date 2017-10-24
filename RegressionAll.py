from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(0, 1, 100) # Обучающая выборка
y = 20 * (np.sin(6 * x * np.pi) + 5 * np.exp(x)) # Некоторая функция
error = 20 * np.random.randn(100) # Ошибка
t = y + error # Целевые значения - некоторая функция + шум
def value_w(F):                     # w = (Ф^T * Ф)^(-1) * (Ф^T * t)
    w = np.dot(F.transpose(), F)    # (Ф^T * Ф)
    w = linalg.inv(w)               # (Ф^T * Ф)^(-1)
    w = np.dot(w, F.transpose())    # (Ф^T * Ф)^(-1) * Ф^T
    w = np.dot(w, t)                # (Ф^T * Ф)^(-1) * (Ф^T * t)
    return w
def value_F(x, n):                  # при n = 1 лин. регрессия
    F=np.zeros((x.size, n + 1))     # при n > 1 полином. регрессия
    for i in range(0, x.size):
        for j in range(0, (n + 1)):
           F[:, j] = pow(x, j)
    return F
def value_z(x, n):                  # z = F * w = value_F(x, n) * value_w(F) = value_F(x, n) * value_w(value_F(x, n)) 
    z = np.dot(value_F(x, n), value_w(value_F(x, n)))
    return z
# График линейной регрессии
plt.figure() # Формируем графическое окно
plt.title('Линейная регрессия')
plt.plot(x, t, '.g') # Целевые значения
plt.plot(x, y , 'r') # Истинный тренд
plt.plot(x, value_z(x, 1), 'b') # Предсказанные значения
#График полиномиальной регрессии
plt.figure() # Формируем графическое окно
plt.title('Полиномиальная регрессия')
plt.plot(x, t, '.g') # Целевые значения
plt.plot(x, y , 'r') # Истинный тренд
plt.plot(x, value_z(x, 9), 'b') # Предсказанные значения
# График зависимости ошибки от степени M
error_sum = []
for i in range(2,20):
    z_polynomial = value_z(x, i)
    error_sum.append(.5 * np.sum((z_polynomial - t)**2))
polynom_powers = np.arange(2,20) # Степени полинома
plt.figure()
plt.title('График зависимости ошибки от степени M') 
plt.plot(polynom_powers,error_sum)