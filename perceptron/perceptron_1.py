import numpy as np
class Perceptron():
    def __init__(self,lr = 0.01):
        self.lr = lr

    def fitting(self,X_train,Y_train,epochs=10):
        X_train = np.hstack((np.ones((X_train.shape[0],1)),X_train))
        self.W = np.zeros((X_train.shape[1],1))
        epoch = 0
        while (epoch < epochs):
            epoch += 1
            isCorrect = self.evaluate(X_train,Y_train,data_extend=True)
            if np.sum(isCorrect)==isCorrect.shape[0]:
                    break
            idx = np.logical_not(isCorrect).reshape(X_train.shape[0]) #切换成一维数组，X_error才能正常赋值
            Y_error = Y_train[idx]
            X_error = X_train[idx,:]
            self.W += self.lr*np.dot(X_error.T,Y_error)

    def predict(self,X_test,data_extend=False):
        if data_extend is False:
            X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
        Y_pred = np.dot(X_test,self.W)
        Y_pred[Y_pred>0]=1
        Y_pred[Y_pred<=0]=-1
        return Y_pred

    def evaluate(self,X_test,Y_test,data_extend=False):
        Y_pred = self.predict(X_test,data_extend)
        result = np.multiply(Y_pred,Y_test)
        return result == 1
    
    def getWeight(self):
        return self.W

def normalization(data):
    m_max = np.max(data, axis=0)
    m_min = np.min(data, axis=0)
    data = (data - m_min)/(m_max-m_min)
    return data

def centering(data):
    m_mean = np.mean(data, axis=0)
    data = data - m_mean
    return data

def preprocessing(data):
    return centering(normalization(data))


data = np.loadtxt("D:/ML_Implementation/Dataset/Breast_Cancer_Coimbra/dataR2.csv",delimiter=",",skiprows=1)
X = data[:,0:9].astype(np.float)
Y = data[:,9]

X = preprocessing(X)
Y[Y==2] = -1
Y = Y.reshape(Y.shape[0],1)
model = Perceptron(0.01)
model.fitting(X,Y,10000)
print(model.getWeight())


