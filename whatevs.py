from numpy import linspace
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

a= (3,4)
b= (6,8)


y_int = cumtrapz([4,8,13],[3,6,9])
print(y_int)
# plt.plot(x, y_int)
# plt.show()