import numpy as np
import pandas as pd

class SoftMax():
	def __init__(self,alpha = 1.0,maxIteration = 1000):
		self.maxIteration = int(maxIteration)
		self.alpha=float(alpha)

	def fit(self,X_,y):
		X = np.hstack((np.ones((X_.shape[0],1)),X_))
		m,n = X.shape
		k=y.shape[1]
		self.theta = np.ones((n,k))
		for i in range(self.maxIteration):
			theta_prev = np.copy(self.theta)
			Dtheta = -X.T@(y-self.softmax(X,self.theta))/m
			self.theta -= self.alpha*Dtheta
			if np.allclose(self.theta, theta_prev):
				break
		acc = self.acc(self.predict(X),np.argmax(y,axis=1))
		print("Iteration:{}".format(i))
		print('train_accuracy:{}'.format(acc))


	def softmax(self,X,theta):
		temp = X @ theta
		return temp/np.sum(temp,axis=1).reshape(-1,1)


	def predict(self,X_):
		X = np.hstack((np.ones((X_.shape[0],1)),X_))
		return np.argmax(self.softmax(X,self.theta),axis=1).reshape(-1,1)
	def acc(self,y,y_pred):
		return (1-np.count_nounzero(y-y_pred))/y.shape[0]
