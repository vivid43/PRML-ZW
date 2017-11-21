#coding = utf8
import numpy as np
import pandas as pd
class LinearRegression():
 	def __init__(self,alpha=1,epsilon=0.001,maxIteration=1000):
 		self.alpha = float(alpha)
 		self.epsilon = float(epsilon)
 		self.maxIteration = int(maxIteration)

 	def fit(self, X_,y):
 		X =  np.hstack(np.ones((X_.shape[0],1)),X_)
 		m,n = X.shape
 		self.theta = np.zeros((n,1))
 		for i in range(maxIteration):
 			Dtheta = X.T @ ( X @ self.theta-y)/m
 			self.theta -= alpha*Dtheta
 			if i % 50 ==0:
 				square_loss = np.sum(np.square(X @ self.theta -y))/2.0/m
 				print("the {} iteration: the loss is ")
 			if np.linalg.norm(Dtheta) <= self.epsilon:
 				break
 	def predict(self,X_):
 		X = np.hstack(np.ones(X_.shape[0],1),X_)
 		return X @ self.theta


