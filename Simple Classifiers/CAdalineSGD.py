#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:05:59 2018

@author: Vitalij Postavnichij
"""

from _funx import X, y, X_std
from numpy.random import seed

class AdalineSGD(object):
    """ Классификатор на основе ADALINE (ADAptive LInear NEuron)
        
        Параметры
        ---------
        eta    : float - Темп обучения
        n_iter : int   - Проходы по тренировочному набору
        
        Атрибуты
        --------
        w_      : array[] - Веса после подгонки
        errors_ : list    - Число случаев ошибочной классификации в эпохе
        shuffle : bool (default:True)
            (true)? Перемешивает тренировочные данные в каждой эпохе,
                    для предотвращения зацикливания
        random_state : int (default:None)
            Инициализирует генератор случайных числе для
            перемешивания и инициализации весов
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    
    def fit(self, X, y):
        """ Выполнить подгонку под тренировочные данные.
    
            Параметры
            ---------
            X : array_like, form = [n_samples, n_features]
                Тренировочные векторы, где
                n_samples  - число образцов и
                n_features - число признаков
            y : array_like, form = [n_samples]
                Целевые значения
            
            Возвращает
            ----------
            self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """ Выполнить подгонку под тренировочные данные
            без повторной инициализации весов """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """ Перемешать тренировочные данные """
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """ Инициализировать веса нулями """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """ Применить обучающее правило ADALINE, чтобы обновить веса  """
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]  += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
       """ Рассчитать читый вход """
       return np.dot(X, self.w_[1:]) + self.w_[0]
   
    def activation(self, X):
        """ Рассчитать линейную активацию """
        return self.net_input(X)
    
    def predict(self, X):
        """ Вернуть метку класса после единичного скачка """
        return np.where(self.activation(X) >= 0.0, 1, -1)

###############################################################################

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

# При динамическом обновлении модели используем
ada.partial_fit(X_std[1:5], y[1:5])

plot_decision_regions(X_std, y, classifier=ada)
plt.title('ADALINE (стохастический градиентный спуск)')
plt.xlabel('длина чашелистника [стандартизованная]')
plt.ylabel('длина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show()

###############################################################################

plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Средняя стоимость')
plt.show()

print(X,y)