#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOGISTIC REGRESSION
Created on Sat Feb 17 10:22:53 2018
@author: Vitalij Postavnichij
"""

from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Разделяем массивы X и y произвольно на тестовые (30%) и тренировочные (70%) данные
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Выполним стандартизацию признаков
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # Вычисляем эмпирическое среднее и стандартное отклонение
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

###############################################################################

# Функция построения графикя областей решения
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap    = ListedColormap(colors[:len(np.unique(y))])
    
    # Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Показать все ообразцы
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, 
                    c=cmap(idx), marker=markers[idx], label=cl)
        # Выделим тестовые образцы
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='black', alpha=.15,
                        linewidths=2, marker='o', s=55, label='тестовый набор')
    
###############################################################################

# Создаем классификатор типа Логическая Регрессия            
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=0)
lr.fit(X_train_std, y_train)
            
###############################################################################            

# Выводим график обастей решения
import matplotlib.pyplot as plt
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined     = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, 
                      test_idx=range(105,150))
plt.title('Логистическая регрессия')
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()

# Оценим вероятность принадлежности образца тестового набора
a = lr.predict_proba(X_test_std[2:3])[0]
print('Вероятность принадлежности первого образца из тестового набора:')
print('Setosa: %.3f, Versicolor: %.3f, Virginica: %.3f' % (a[0], a[1], a[2]))

###############################################################################

# Построим график траектории L2-регуляризации для двух весовых коэфициентов
weights, params = [], []
for c in np.arange(-5.0, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:,0], label='длина лепестка')
plt.plot(params, weights[:,1], linestyle='--', label='ширина лепестка')
plt.ylabel('весовой коэффициент')
plt.xlabel('C')
plt.title('Траектория L2-регуляризации для двух весовых коэф.')
plt.xscale('log')
plt.grid()
plt.show()















