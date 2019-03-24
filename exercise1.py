# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = np.loadtxt("dataForTesting.txt")       # 将txt文本存为numpy数组
data_train = np.loadtxt("dataFortraining.txt")       # 将txt文本存为numpy数组

m, n = np.shape(dataset)
M, N = np.shape(data_train)

x, x_train = np.ones((m, n)), np.ones((M, N))

x[:, : -1], x_train[:, : -1] = dataset[:, : -1],data_train[:, : -1]   #获得（x0，x1，x2）
y, y_train = dataset[:, -1], data_train[:, -1]        #获得Y
MAX_Inter = 150000
precision = 0.0001
step, s = 0.00015, 5000
err = [0, 0, 0]
error_train, error, num, Iter = 0,0,0,0
theta = [1, 1, 1]
loss, loss_train = 10, 10
Error, Error_train = [0]*30,  [0]*30
while (Iter < MAX_Inter and loss > precision):
    loss = 0
    loss_train = 0
    if (num > m - 1):  # 梯度下降方法
        num = 0
    # num = random.randint(0, m - 1)  # 随机梯度下降方法
    prediction = theta[0] * x[num][0] + theta[1] * x[num][1] + theta[2] * x[num][2]
    err[0] = (prediction - y[num]) * x[num][0]
    err[1] = (prediction - y[num]) * x[num][1]
    err[2] = (prediction - y[num]) * x[num][2]
    num += 1
    for index in range(3):
        theta[index] = theta[index] - step * err[index]
    for index in range(m):     #计算测试误差
        prediction = theta[0] * x[index][0] + theta[1] * x[index][1] + theta[2] * x[index][2]
        error = (1 / 2/m) * (prediction - y[index]) ** 2    #loss
        loss += error
    for index in range(M):   #计算训练方差
        prediction_train = theta[0] * x_train[index][0] + theta[1] * x_train[index][1] + theta[2] * x_train[index][2]
        error_train = (1 / 2/M) * (prediction_train - y_train[index]) ** 2    #loss
        loss_train +=error_train
    Iter = Iter + 1

    if(Iter == s):
        k = int(s/5000)
        Error[k-1] = error
        Error_train[k-1] = error_train
        print("theta", theta)
        s = s+5000  #每隔10000输出一次
print('Error',Error)
print('Error_train', Error_train)
X  = [i*5000for  i in range(30)]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(X,Error)
ax.set_title('Loss_train')
plt.savefig(r"Loss_train.png")
fig = plt.figure()
bx = fig.add_subplot(1, 1, 1)
bx.plot(X,Error_train)
bx.set_title('Loss_test')
plt.savefig(r"Loss_test.png")

