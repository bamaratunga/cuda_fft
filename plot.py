#! /usr/bin/python3
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


data = genfromtxt('output_0.csv', delimiter=',')
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
data = np.fft.fftshift(data)
plt.imshow(data, extent=[-1, 1, -1, 1])
plt.show()
