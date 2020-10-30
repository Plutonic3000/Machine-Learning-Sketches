#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAptive LInear NEuron (ADALINE) class
  with batch gradient descent (GD)
Created on Wed Feb 14 17:55:37 2018
@author: Vitalij Postavnichiy
"""

import _funx
    

class AdalineGD(object):
    """ Классификатор на основе ADALINE
        
        Параметры
        ---------
        eta    : float - Темп обучения (между 0.0 и 1.0)
        n_iter : int - Проходы по тренировочному набору
        
        Атрибуты
        --------
        w_      : Веса после подгонки
        errors_ : Число случаев ошибочной классификации
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """ Выполнить подгонку под тренировочные данные
            Параметры
            ---------
            X : array_like, form = [n_samples, n_features]
                - тренировочные векторы, где 
                  n_samples  - число образцов
                  n_features - число признаков
            y : array_like, form = [n_samples]
                - целевые значения
            
            Возвращает
            ----------
            self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X): # Рассчитать чистый вход
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X): # Рассчитать линейную активацию
        return self.net_input(X)
    
    def predict(self, X): # Вернуть метку класса после единичного скачка
        return np.where(self.activation(X) >= 0.0, 1, -1)

###############################################################################
# Строим график стоимость/эпохи
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

# C темпом обучения 0.01
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Сумма квадратичных ошибок)')
ax[0].set_title('ADALINE (темп обучения 0.01)')

# С темпом обучения 0.0001
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('log(Сумма квадратичных ошибок)')
ax[1].set_title('ADALINE (темп обучения 0.0001)')
plt.show()    

###############################################################################

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Адалин (градиентный спуск)')
plt.xlabel('длина чашелистника [стандартизованная]')
plt.ylabel('длина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.show()
    
###############################################################################    
    
    
    
    
    
    
    
    
    
        