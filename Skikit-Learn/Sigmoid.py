#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIGMOID FUNCTION
Created on Sat Feb 17 09:13:49 2018
@author: Vitalij Postavnichij
"""

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks(np.arange(0, 1.25, .25))
plt.grid()
plt.ylim(-0.1, 1.1)
plt.title('Сигмоида')
plt.xlabel('$z$')
plt.ylabel('$\phi (z)$')
plt.show()

