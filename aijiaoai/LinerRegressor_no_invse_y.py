#coding:utf-8

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler
import numpy as np
data,y=load_boston(return_X_y=True)
# print(list(load_boston().data))
X_train=data[:450]
y_train=y[:450]
X_test=data[450:]
y_test=y[450:]

class LinerRegressor:

    def __init__(self,alpha=0.001,epoch=1000):
        self.alpha=alpha
        self.epoch=epoch

    def preProssData(self,X,y=None,is_train=True,is_stand=False):
        if is_train:
            # 标准化X
            if is_stand:
                stand = StandardScaler()
                X = stand.fit_transform(X)
                self.standX = stand
            else:
                minMax = MinMaxScaler()
                X = minMax.fit_transform(X)
                self.minMax = minMax
            # onehot
            oneHot1 = OneHotEncoder()
            X_col = oneHot1.fit_transform(X[:, 3].reshape(-1, 1))
            self.oneHot1 = oneHot1
            X = np.delete(X, [3], axis=1)
            X = np.concatenate((X, X_col.toarray()), axis=1)
        else:
            # 标准化X
            if is_stand:
                X = self.standX.transform(X)
            else:
                X = self.minMax.transform(X)
            # onehot
            oneHot1 = self.oneHot1
            X_col = oneHot1.transform(X[:, 3].reshape(-1, 1))
            X = np.delete(X, [3], axis=1)
            X = np.concatenate((X, X_col.toarray()), axis=1)
        return X, y

    def fit(self,X,y,is_stand=False):
        X,y=self.preProssData(X,y,is_train=True,is_stand=is_stand)
        ones=np.ones([len(X),1])
        X=np.concatenate((X,ones),axis=1)
        theta=np.random.rand(X.shape[1])
        self.theta = theta
        for _ in range(self.epoch):
            index=np.random.randint(0,len(X))
            xi,yi =list(zip(X,y))[index]
            grad=self.sgd_optimiz(xi,yi)
            theta=theta-self.alpha*grad

    #梯度
    def sgd_optimiz(self,xi,yi):
        return (xi.dot(self.theta)-yi)*xi

    def score(self,X,y,is_stand=False):
        X,y=self.preProssData(X,y,is_train=False,is_stand=is_stand)
        ones = np.ones([len(X), 1])
        X = np.concatenate((X, ones), axis=1)
        res=(X.dot(self.theta).reshape(-1,1))
        return sum(1-np.abs(res.reshape(-1)-y.reshape(-1))/y.reshape(-1))/len(y)

if __name__=='__main__':
    regressor=LinerRegressor(alpha=0.01,epoch=100000)
    regressor.fit(X_train,y_train,is_stand=True)
    print(regressor.score(X_test,y_test,is_stand=True))
    # print(X_train[:,3])