import numpy as np
import matplotlib.pyplot as plt

length = 10
xTrain = np.arange(length)
yTrain = np.random.randint(0,2,length)

w,b = 0,0
learningRate = 1e-2
iterations = 1000
for i in range(iterations):
    dw = (1/length)*np.dot(w*xTrain+b-yTrain,xTrain.T)
    db = (1/length)*np.sum((w*xTrain+b-yTrain))
    w = w - learningRate*dw
    b = b - learningRate*db
plt.scatter(xTrain,yTrain,label='Values')
plt.plot(xTrain,w*xTrain+b,'r',label=f'$ y={w:.2}x+{b:.2}$')
plt.legend()
plt.show()