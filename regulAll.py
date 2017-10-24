from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt

# Design matrix
def computing_F_polynomial(x,n):
    F=np.ones((x.size, n+1))
    for i in range(1,n+1):
        F[:,i]=x**i
    return F

# Веса
def computing_w(F,t,lamda):
    I=np.identity(F.shape[1])
    return linalg.inv((F.transpose().dot(F))+lamda*I).dot(F.transpose()).dot(t)

# Ищем лучшую лямбду на validation set
def find_lambda(lambda_array,x_val,t_val):
    matr_val = computing_F_polynomial(x_val,9)
    error = []
    
    for i in range(0,len(lambda_array) - 1):
        w_polynomial = computing_w(matr_val,t_val,lambda_array[0])
        z_polynomial = matr_val.dot(w_polynomial)
        error.append(.5 * np.sum((z_polynomial - t_val)**2))
    
    return error.index(min(error))

# Регуляризация
x = np.linspace(0, 1, 1000) # Весь data set

lambda_array = [0.001,0.01,0.1,1,10,50,100,500,1000]

y = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x) # Некоторая функция
error = 10 * np.random.randn(1000) # Ошибка
t = y + error # Целевые значения - некоторая функция + шум

ind = np.arange(1000)
np.random.shuffle(ind)

x_train = x[ind[:600]]
x_val = x[ind[600:800]]
x_test = x[ind[800:]]

t_train = t[ind[:600]]
t_val = t[ind[600:800]]

y_train = y[ind[:600]]
y_val = y[ind[600:800]]
y_test = y[ind[800:]]

# Полиномиальная регрессия
matr = computing_F_polynomial(x_train,9) # Из предыдущего задания вычислил наилучшую степень

# Здесь присоединяем к имеющимся столбцам sin,cos,exp,на основе training set выбираем лучшие веса
# Но так до конца и не понял как это делать
#new = np.zeros((600,1))
#new[:,0] = np.sin(x_train)
#new[:,1] = np.cos(x_train)
#new[:,2] = np.exp(x_train)
#matr = np.hstack((matr,new))

w_polynomial = computing_w(matr,t_train,0)
z_polynomial = matr.dot(w_polynomial)

# График
plt.figure() # Формируем графическое окно
plt.title('Полиномиальная регрессия на training set')

plt.plot(x_train, t_train, '.g') # Целевые значения
plt.plot(x_train, y_train , 'r',  marker='.',linestyle='none') # Истинный тренд
plt.plot(x_train, z_polynomial, 'b', marker='.',linestyle='none') # Предсказанные значения

best_lambda = lambda_array[find_lambda(lambda_array,x_val,t_val)]
w_polynomial = computing_w(matr,t_train,best_lambda)

# Проверка на test set
y_predicted = computing_new_y(w_polynomial,x_test) # На основе полученной модели вычисляем результат

# График
plt.figure() # Формируем графическое окно
plt.title('Проверка на test set')

plt.plot(x_test, y_test , 'r',  marker='.',linestyle='none') # Истинный тренд
plt.plot(x_test, y_predicted, 'b', marker='.',linestyle='none') # Предсказанные значения