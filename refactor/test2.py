import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, m):
    y = m*x
    return y

x = [0.2, 0.4, 0.5, 1, 1.5, 2, 2.5]
y = [4.41959521, 8.83919042, 11.04898802, 22.09797605, 33.14696407, 44.1959521, 55.24494012]

params, covars = curve_fit(linear, x, y)
print(params)

plt.scatter(x, y)
plt.plot(x, linear(np.array(x), *params))
plt.show()