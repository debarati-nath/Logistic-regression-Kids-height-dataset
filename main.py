#Data Upload and Visualisation:
#Import libraries
from scipy import optimize
from matplotlib import pyplot
import os
import numpy as np
import pandas as pd
from scipy import optimize
import sys

#Load data
data=pd.read_csv('C:/Users/Debarati/Desktop/KidsHeightData.csv')
X= data.iloc[:,0:2].values
y= data.iloc[:,2].values

#Visualization
def plotData(X, y):
fig = pyplot.figure()
fig.add_axes([0,0,2,2])
pos=y==1
neg = y == 0
pyplot.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=5, mec='k', mew=1)
plotData(X, y)
pyplot.xticks(np.arange(0,9,step=1))
pyplot.yticks(np.arange(4.5,8.5,step=0.5))
pyplot.xlabel('Shoe Size at Age 3')
pyplot.ylabel('Average Height of Parents')
pyplot.legend(['Kids who grew up to be >=2m', 'Kids who grew up to < 2m'])
pyplot.title("Training data for the height prediction problem")
pass

#Sigmoid function
def sigmoid(z):
z = np.array(z)
g = np.zeros(z.shape)
g = 1 / (1 + np.exp(-z))
return g

#Logistic regression
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def costFunction(theta, X, y):
m = y.size
J = 0
grad = np.zeros(theta.shape)
h = sigmoid(X.dot(theta.T))
J = ((1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))))
grad = (1 / m) * (h - y).dot(X)
return J, grad
initial_theta = np.random.randn(n+1) #initial_theta = np.zeros(n+1)
cost, grad = costFunction(initial_theta, X, y)
print('Cost at initial theta: {:.3f}'.format(cost))
print('Gradient at initial theta:')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))

#Minimization function
options= {'maxiter': 700}
res = optimize.minimize(costFunction,initial_theta,(X, y),jac=True,method='TNC',options=options)
cost = res.fun
theta = res.x
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))

#Decision boundary
def mapFeature(X1, X2, degree=7):
if X1.ndim > 0:
out = [np.ones(X1.shape[0])]
else:
out = [np.ones(1)]
for i in range(1, degree + 1):
for j in range(i + 1):
out.append((X1 ** (i - j)) * (X2 ** j))
if X1.ndim > 0:
return np.stack(out, axis=1)
else:
return np.array(out)

#Plot decision boundary
def plotDecisionBoundary(plotData, theta, X, y):
theta = np.array(theta)
plotData(X[:, 1:3], y)
if X.shape[1] <= 3:
# Only need 2 points to define a line, so choose two endpoints
plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
pyplot.plot(plot_x, plot_y)
# Legend, specific for the exercise
pyplot.legend(['Kids who grew up to be >=2m', 'Kids who grew up to < 2m', 'Decision Boundary'])
pyplot.xlim([1, 8])
pyplot.ylim([4, 8])
else:
u = np.linspace(-3, 5, 1)
v = np.linspace(-3, 5, 1)
z = np.zeros((u.size, v.size))
for i, ui in enumerate(u):
for j, vj in enumerate(v):
z[i, j] = np.dot(mapFeature(ui, vj), theta)
z = z.T # important to transpose z before calling contour
pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)
plotDecisionBoundary(plotData, theta, X, y)

#Prediction
def predict(theta, X):
m = X.shape[0]
p = np.zeros(m)
p = np.round(sigmoid(X.dot(theta.T)))
return p
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))

