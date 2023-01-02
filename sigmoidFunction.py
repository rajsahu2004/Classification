import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-50,50)
y = 1/(1+np.exp(-x))
print(y)
plt.plot(x,y)
plt.show()