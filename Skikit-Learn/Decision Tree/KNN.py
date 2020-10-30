#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-nearest neighbor classifier, KNN
Created on Thu Feb 22 11:33:45 2018
@author: Vitalij Postavnichij
"""
# Расширяем область видимости модуля на предыдущую папку для импорта
import sys
sys.path.append('../')
from _funx import plot_decision_regions, X_train_std, y_train, X_test, y_test
from _funx import X_train, y_train, X_combined_std, y_combined

import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=17, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn,
                      test_idx=range(105,150))
plt.xlabel('длина лепестка (стандартизованная)')
plt.ylabel('ширина лепестка (стандартизованная)')
plt.show()
