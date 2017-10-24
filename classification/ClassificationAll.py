import numpy as np

def α_error(FalsePositive, TrueNegative):
    return round((FalsePositive / (TrueNegative + FalsePositive)), 3)

def β_error(FalseNegative, TruePositive):
    return round((FalseNegative/(TruePositive + FalseNegative)), 3)

def accuracy(TruePositive, TrueNegative, N):
    return round(((TruePositive + TrueNegative) / N), 3)

def precision(TruePositive, FalsePositive):
    return round((TruePositive / (TruePositive + FalsePositive)), 3)

def recall(TruePositive, FalseNegative):
    return round((TruePositive/(TruePositive + FalseNegative)), 3)

def F1_Score(Precision, Recall):
    return round((2 * (Precision * Recall) / (Precision + Recall)), 3)

f_growth        = np.random.normal(180, 7, 500)                         
b_growth        = np.random.normal(200, 7, 500) 
all_rowth       = np.concatenate((f_growth, b_growth))                                   
coin            = []
value_coin      = [0, 0, 0, 0]     # [TN, FN, FP, TP]
value_threshold = [0, 0, 0, 0]     # [TN, FN, FP, TP]
# 1 ый способ - монетка
def coint_way():
    for i in range(0, 1000):
        coin.append(np.random.randint(0, 2))
        if i < 500:
            if coin[i] == 0:
                value_coin[0] += 1
            else:
                value_coin[1] += 1
        else:
            if coin[i] == 0:
                value_coin[2] += 1
            else:
                value_coin[3] += 1
    return value_coin
value_coin = coint_way()
# 2 ой способ - порог
def threshold_way(): 
    for i in range(0,1000):
        if i < 500:
            if all_rowth[i] <= 195:
                value_threshold[0] += 1
            else:
                value_threshold[1] += 1
        else:
            if all_rowth[i] <= 195:
                value_threshold[2] += 1
            else:
                value_threshold[3] += 1
    return value_threshold
value_threshold = threshold_way()

α_coin              = α_error(value_coin[2],        value_coin[0])                                                     # Ошибка 1 рода для монетки
α_threshold         = α_error(value_threshold[2],   value_threshold[0])                                                # Ошибка 1 рода для порогоа
β_coin              = β_error(value_coin[1],        value_coin[3])                                                     # Ошибка 2 рода для монетки
β_threshold         = β_error(value_threshold[1],   value_threshold[3])                                                # Ошибка 2 рода для порога
accuracy_coin       = accuracy(value_coin[3],       value_coin[0],      footballers.size + basketball_players.size)    # Accuracy для монетки
accuracy_threshold  = accuracy(value_threshold[3],  value_threshold[0], footballers.size + basketball_players.size)    # Accuracy для порога
precision_coin      = precision(value_coin[3],      value_coin[2])                                                     # Precision для монетки
precision_threshold = precision(value_threshold[3], value_threshold[2])                                                # Precision для порога
recall_coin         = recall(value_coin[3],         value_coin[1])                                                     # Recall для монетки
recall_threshold    = recall(value_threshold[3],    value_threshold[1])                                                # Recall для порога
F1_Score_coin       = F1_Score(precision_coin,      recall_coin)                                                       # F1 Score для монетки
F1_Score_threshold  = F1_Score(precision_threshold, recall_threshold)                                                  # F1 Score для порога
print('-----------------------------------------------------')
print('Ошибка 1 рода для монетки    = ', α_coin)
print('Ошибка 1 рода для порога     = ', α_threshold)
print('-----------------------------------------------------')
print('Ошибка 2 рода для монетки    = ', β_coin)
print('Ошибка 2 рода для порога     = ', β_threshold)
print('-----------------------------------------------------')
print('Accuracy для монетки         = ', accuracy_coin)
print('Accuracy для порога          = ', accuracy_threshold)
print('-----------------------------------------------------')
print('Precision для монетки        = ', precision_coin)
print('Precision для порога         = ', precision_threshold)
print('-----------------------------------------------------')
print('Recall для монетки           = ', recall_coin)
print('Recall для порога            = ', recall_threshold)
print('-----------------------------------------------------')
print('F1 Score для монетки         = ', F1_Score_coin)
print('F1 Score для порога          = ', F1_Score_threshold)