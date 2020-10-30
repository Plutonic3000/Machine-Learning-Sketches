#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реализация функции ядра радиального базиса (RBF)
Created on Sat Feb 17 14:02:18 2018
@author: Vitalij Postavnichij
"""
# Расширяем область видимости модулей на предыдущую папку
import sys
sys.path.append('../')
from _funx import plot_decision_regions, X_train_std, y_train
from _funx import X_train, y_train, X_combined_std, y_combined

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

"""
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], 
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], 
            c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()
"""

###############################################################################

"""
# Нелинейное разделение двух классов

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.40, C=1.20)
# gamma и C - изменяют строгость дифференцирования данных
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
"""

###############################################################################

# Применим функции радиального базиса к набору Iris

#from _funx import X_train, y_train, X_combined_std, y_combined

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                      test_idx=range(105,150))
plt.xlabel('длина чашелистника (стандартизованная)')
plt.ylabel('ширина лепестка (стандартизованная)')
plt.legend(loc='upper left')
plt.show()
