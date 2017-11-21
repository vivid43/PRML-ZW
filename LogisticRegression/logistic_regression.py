
# coding: utf-8

# 不加正则化项的logistic回归,代码进行了简化处理

import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, alpha=1.0, epsilon=0.01, maxIteration=1000):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.maxIteration = int(maxIteration)
        self.theta = None

    def fit(self, X_, Y):
        m = X_.shape[0]
        n = X_.shape[1] + 1
        X = np.hstack((np.ones((m,1)), X_))
        self.theta = np.zeros((n, 1))
        for i in range(self.maxIteration):
            dTheta = -(X.T @ (Y - self.sigmoid(X))) / m
            self.theta -= self.alpha * dTheta
            if np.linalg.norm(dTheta) <= self.epsilon:
                break
        acc = 1 - np.count_nonzero(Y - self.predict(X_)) / m
        print(acc, i)

    def sigmoid(self, X):
        return 1. / (1. + np.exp(-(X @ self.theta)))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        return (self.sigmoid(X) > 0.5) + 0