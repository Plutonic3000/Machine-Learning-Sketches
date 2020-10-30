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

###############################################################################

# Применим функции радиального базиса к набору Iris

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
# gamma - определяет строгость границы решений
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                      test_idx=range(105,150))
plt.xlabel('длина чашелистника (стандартизованная)')
plt.ylabel('ширина лепестка (стандартизованная)')
plt.legend(loc='upper left')
plt.show()
