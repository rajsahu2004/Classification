import numpy as np
import matplotlib.pyplot as plt
def predict(x,w,b):
    z = np.dot(x,w)+b
    return 1/(1+np.exp(-z))
m = 50
xTrain = np.arange(m)
yTrain = np.sort(np.random.randint(0,2,m))
w,b = 0,0
learningRate,iterations = 1e-2,100000
costList = np.array([])
for i in range(iterations):
    cost = (1/(2*m))*np.sum(np.square(predict(xTrain,w,b)-yTrain))
    costList = np.append(costList,cost)
    dw = (1/m)*np.dot(predict(xTrain,w,b)-yTrain,xTrain.T)
    db = (1/m)*np.sum(predict(xTrain,w,b)-yTrain)
    w = w - learningRate * dw
    b = b - learningRate * db
    if i%(iterations/20)==0:print(f'{i}. Cost = {cost}\bw ={b}\tw = {w}')
fig,axes = plt.subplots(1,2,figsize=(12,6))
axes[0].scatter(xTrain,yTrain,label='points')
axes[0].plot(xTrain,predict(xTrain,w,b))
axes[1].plot(costList)
fig.tight_layout()
plt.show()