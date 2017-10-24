from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

x = np.linspace(0, 1, 1000)

ind = np.arange(1000)
np.random.shuffle(ind)

# Первая функция
x_1 = x[ind[:500]]
y_1 = (20 * np.sin(2 * np.pi * 3 * x_1) + 100 * np.exp(x_1) + 80) / 120 # Некоторая функция
error = (10 * np.random.randn(500)) / 100 # Ошибка
t_1 = y_1 + error

# Добавляем класс
class_0 = np.zeros(shape=(t_1.size))
x1_t1 = np.vstack((x_1,t_1)).transpose()
x1_t1_with_class = np.column_stack((x1_t1, class_0))

# Вторая функция
x_2 = x[ind[500:1000]]
y_2 = (20 * np.sin(2 * np.pi * 3 * x_2) + 100 * np.exp(x_2) ) / 120 # Некоторая функция
error = (10 * np.random.randn(500)) / 100 # Ошибка
t_2 = y_2 + error

# Добавляем класс
class_1 = np.ones(shape=(t_2.size))
x2_t2 = np.vstack((x_2,t_2)).transpose()
x2_t2_with_class = np.column_stack((x2_t2, class_1))

x_y_data_set = np.vstack((x1_t1_with_class, x2_t2_with_class)) # Весь data set

# Разделяем выборку
ind = np.arange(1000)
np.random.shuffle(ind)

x_y_train = x_y_data_set[ind[:600]]
x_y_val = x_y_data_set[ind[600:800]]
x_y_test = x_y_data_set[ind[800:]]

def computing_fert(x_y,m,fert_indexes):
    fert = np.ones((1, m+1))
    
    # Все возможные столбцы матрицы ферт,выбираем случайно 6 из них
    fert_all = np.zeros((1,10))
    fert_all[0,0] = x_y[0] 
    fert_all[0,1] = x_y[1]
    fert_all[0,2] = np.sin(x_y[0])
    fert_all[0,3] = np.sin(x_y[1])
    fert_all[0,4] = np.cos(x_y[0])
    fert_all[0,5] = np.cos(x_y[1])
    fert_all[0,6] = 0.5 * x_y[0]
    fert_all[0,7] = 0.5 * x_y[1]
    fert_all[0,8] = 1.2 * x_y[1]
    fert_all[0,9] = 1.2 * x_y[0]
    
    for i in range(1,m+1):
        fert[0,i] = fert_all[0,fert_indexes[i - 1]]
        
    return fert.transpose()

def sigmoid(x_y,w,m,fert_indexes):
    exp = -w.dot(computing_fert(x_y,m,fert_indexes))
    sig = 1 / (1 + np.power(np.e,exp))
    
    return sig.transpose()

def gradient(N,t,w,lam,x_y,m,fert_indexes):
    inter = 0
    for i in range(0,N):
        inter += np.asscalar(sigmoid(x_y[i],w,m,fert_indexes) - t[i]) * computing_fert(x_y[i],m,fert_indexes)
        
    grad = inter.transpose() + lam * w
                      
    return grad
   
def gradient_descent(N,t,x_y_train,lam,fert_indexes):
    w_0 = np.matrix([[0.01,0.001,0.1,0.1,0.01,0.01,0.1]])
    w_k_1 = w_0
    w_k = w_k_1 - 0.01 * gradient(N,t,w_k_1,lam,x_y_train[:,0:2],6,fert_indexes)
    while True:
        w_k = w_k_1 - 0.01 * gradient(N,t,w_k_1,lam,x_y_train[:,0:2],6,fert_indexes)      
        if linalg.norm(w_k - w_k_1) < 0.01 * (linalg.norm(w_k) + 0.001):
            break
        else:
            w_k_1 = w_k
            
    return w_k

def error(w,t,N,x_y,fert_indexes):
    result = 0
    for i in range(0,N):
        result += t[i] * np.log(sigmoid(x_y[i],w,6,fert_indexes)) + (1 - t[i]) * np.log(1 - sigmoid(x_y[i],w,6,fert_indexes))
    
    return -result

def alpha_error(FP,TN):
    return FP/(TN + FP)

def beta_error(FN,TP):
    return FN/(TP + FN)

def accuracy(TP,TN,N):
    return (TP + TN) / N

def precision(TP,FP):
    return TP / (TP + FP)

def recall(TP,FN):
    return TP/(TP + FN)

def F1_Score(Pr,Rec):
    return 2 * (Pr * Rec) / (Pr + Rec)

# Для построения разделяющей кривой
def f(y):
    value = w[0,0] + w[0,1] * y + w[0,2] * 1.2 * y + w[0,3] * np.cos(x_boundary) + w[0,4] * 1.2 * x_boundary + w[0,5] * np.sin(x_boundary) + w[0,6] * 0.5 * x_boundary 
    return value

# Возможные значения лямбды
lambda_array = [0,0.0001,0.001,0.01,0.1,0.2,0.25,0.3,0.4,0.5]

errors = np.zeros(10)
params_lambda = np.zeros(10)
params_fert = np.zeros((10,6))

#Цикл работает в среднем 30 секунд,но в плохих случаях долго - можно два цикла - отдельный для определения лучшей лямбды
#и отдельный для определения лучшего ферт,но так в два раза дольше,а результат лучше всего на 1 процент
#for i in range(0,10):
#    print('Итерация номер ',i + 1)
#    lambda_random = np.random.randint(len(lambda_array))
#    fert_indexes = np.random.choice(range(10), 6, replace=False) 
#    params_lambda[i] = lambda_array[lambda_random]
#    params_fert[i] = fert_indexes        
#    test_gradient_descent = gradient_descent(x_y_train[:,0].size,x_y_train[:,2:3],x_y_train,lambda_array[lambda_random],fert_indexes)
#    errors[i] = error(test_gradient_descent,x_y_val[:,2:3],x_y_val[:,2:3].size,x_y_val[:,0:2],fert_indexes)

# На случай если будет работать долго, то цикл выше и 3 строки ниже закомментировать
best_lambda = 0.25
best_fert = [1,8,4,9,2,6]

#best_index = errors.argmin(axis=0) # Лучшие столбцы матрицы ферт
#best_lambda = params_lambda[best_index]
#best_fert = params_fert[best_index].astype(int)

best_gradient_descent = gradient_descent(x_y_train[:,0].size,x_y_train[:,2:3],x_y_train,best_lambda,best_fert)

table_result = [[0,0],[0,0]]

result = np.zeros(200)
for i in range(0,200):
    result[i] = sigmoid(x_y_test[i,:2],best_gradient_descent,6,best_fert)

for i in range(0,200):
    if result[i] >= 0.5:
        result[i] = 1
    else:
        result[i] = 0

# Истинный результат
inter = x_y_test[:,2:3]

for i in range(0,x_y_test[:,2:3].size):
    if (inter[i] == 0) and (result[i] == 0):
        table_result[0][0] += 1
    elif (inter[i] == 0) and (result[i] == 1):
        table_result[0][1] += 1
    elif (inter[i] == 1) and (result[i] == 0):
        table_result[1][0] += 1
    elif (inter[i] == 1) and (result[i] == 1):
        table_result[1][1] += 1

alpha_error = alpha_error(table_result[1][0],table_result[0][0])

print()
print('Ошибка первого рода = ',alpha_error)
print()

beta_error = beta_error(table_result[0][1],table_result[1][1])

print('Ошибка второго рода = ',beta_error)
print()

accuracy = accuracy(table_result[1][1],table_result[0][0],result.size)

print('Accuracy = ', accuracy)
print()

precision = precision(table_result[1][1],table_result[1][0])

print('Precision = ', precision)
print()

recall = recall(table_result[1][1], table_result[0][1])

print('Recall = ', recall)
print()

F1_Score = F1_Score(precision, recall)

print('F1 Score = ', F1_Score)

# Построение разделяющие кривой - только для одного наилучшего случая,для всех не успел
w = best_gradient_descent
y_boundary = np.zeros(200)

for i in range(0,200):
    x_boundary = x_y_test[i,0]
    y_boundary[i] = opt.fsolve(f,0)

plt.plot(x_y_test[:,0],x_y_test[:,1],'.g')
plt.plot(x_y_test[:,0],y_boundary,'.r')