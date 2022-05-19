import numpy as np

#目前只支持mse，后续会开发cross entropy
class mse_loss():
    def __init__(self,batch_size):
        self.batch_size=batch_size

    def gradient(self,data_X,data_Y):
        """
        data_X：批量前向运算结果，不是列向量
        data_Y：列向量
        """
        return data_X.T-data_Y
    
    def loss(self,data_X,data_Y):
        temp=self.gradient(data_X,data_Y)
        return np.dot(temp.T,temp)/(2*self.batch_size)