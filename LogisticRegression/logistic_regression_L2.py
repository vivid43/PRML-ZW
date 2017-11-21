import numpy as np
import pandas as pd

'''
logistic回归
1、批梯度下降
2、L2 正则化项
'''
class LogisticRegression():
    def __init__(self,alpha = 1,lmbda =0,maxiter = 1000,epsilon = 0.0001):
        self.alpha = float(alpha) #学习率
        self.lmbda = float(lmbda) #正则化系数
        self.maxiter = int(maxiter)#最大迭代次数
        self.epsilon = 0.001#导数收敛阈值
        self.threshold = 0.5 #分类阈值

    def process_feature(self,X):
        return np.hstack((np.ones(shape=(X.shape[0],1),dtype='float'),X))

    def sigmoid(self,X):
        return 1./(1.+np.exp(-np.dot(X,self.theta)))

    def fit(self,X_,y):
        X = self.process_feature(X_)
        self.m,self.n = X.shape
        self.theta = np.zeros(shape=(self.n,1),dtype = 'float')   
        for i in range(self.maxiter):
            temp = self.theta
            temp[0,0] = 0
            dTheta0 = (self.lmbda/self.m)*temp
            dTheta1 = (1/self.m)*np.dot(X.T,(self.sigmoid(X)-y))
            dTheta = dTheta1+dTheta0
            self.theta -= self.alpha*dTheta
            d_norm = np.linalg.norm(dTheta)
            if i % 100 ==0:
                acc = 1-np.count_nonzero(y-self.predict(X_))/self.m
                print("iteration: {};   Accuracy: {}; Derivate: {}".format(i, acc,d_norm))
            if d_norm<self.epsilon:
                break
        accuracy = 1-np.count_nonzero(y-self.predict(X_))/self.m
        print("Break at the {} iteration: the accuracy of training set is  {}".format(i,accuracy))
        print("the d_norm is {}".format(d_norm))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        return (self.sigmoid(X) > 0.5) + 0