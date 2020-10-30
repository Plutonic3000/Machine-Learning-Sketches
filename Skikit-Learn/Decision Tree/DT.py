#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""
# Построим график индексов неоднородности для трех разных критериев:
# - мера неоднородности Джини
# - энтропия
# - ошибка классификации

def gini(p):
    return (p)*(1-(p))+(1-p)*(1-(1-p))

def entropy(p):
    return - p*np.log2(p)-(1-p)*np.log2((1-p))

def error(p):
    return 1-np.max([p, 1-p])

x = np.arange(0.0, 1.0, .01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                         ['Энтропия','Энтропия(шкалир.)',
                          'Неоднородность Джини', 'Ошибка классификации'],
                         ['-', '-', '--', '-.'],
                         ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Индекс неоднородности')
#plt.show()
"""
###############################################################################
"""
# Классификатор типа дерево
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105,150))
plt.xlabel('длина лепестка (см)')
plt.ylabel('ширина лепестка (см)')
plt.legend(loc='upper left')
plt.show()
"""
###############################################################################
"""
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', 
                feature_names=['petal length','petal width'])
"""
###############################################################################

# Метод случайный лесов (объединение классификаторов типа дерево в группу)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=15,
                                random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105,150))
plt.xlabel('длина лепестка (см)')
plt.ylabel('ширина лепестка (см)')
plt.legend(loc='upper left')
plt.show()

###############################################################################

