import numpy as np
import matplotlib.pyplot as plt
from Loss import mse_loss
from optimizers import SGD_optimizer
from Model import Model, Dense
from dataloader import randDataGenerator, dataset_generator
from config import *

#使用单隐层神经玩过实现线性回归演示
class Linear_regression():
    def __init__(self,dataset_generator,
                learning_rate=0.01,
                epoch=20,
                batch_size=20,
                optimizer=SGD_optimizer(learning_rate=0.01),
                loss="mse"):

        self.model=Model([Dense(dataset_generator.shape[1]-1,1)])
        self.optimizer=optimizer
        self.optimizer.opt(self.model) #设置学习率

        self.dataset=dataset_generator
        self.epoch=epoch
        self.batch_size=batch_size
        self.Loss=[]

        if loss=="mse":
            self.loss=mse_loss(self.batch_size)
        else:
            print("目前只支持mse")
            assert 1==0 #强制终止程序

    def train(self):
        iter=0
        while iter<self.epoch:
            print("epoch"+str(iter)+":")
            np.random.shuffle(self.dataset) #随机打乱数据集顺序，
            Loss_in_epoch=[]
            data_list=np.split(self.dataset,self.dataset.shape[0]/self.batch_size,axis=0)
            for item in data_list:
                dataX=item[:,:-1].T
                dataY=item[:,-1].reshape(self.batch_size,1)
                original_res=self.loss.gradient(self.model.forward(dataX),dataY)
                self.model.backpropagate(original_res=original_res)
                Loss_in_epoch.append(self.loss.loss(self.model.forward(dataX),dataY))
            print(np.mean(Loss_in_epoch))
            self.Loss.append(np.mean(Loss_in_epoch))
            iter=iter+1

    def visualize_loss(self):
        #x=np.linspace(1,self.epoch,self.epoch).astype("int8")
        plt.xlabel("epoches")
        plt.ylabel("Loss")
        plt.plot(self.Loss, 'c-',linewidth=2)
        plt.show()

if __name__ == "__main__":
    data=randDataGenerator(num_sample=num_sample,low=low,high=high)
    dataset=dataset_generator(data,weight=weight,bias=bias,no_zenguang=True) #其中有一个维度为增广维度
    model=Linear_regression(dataset)
    model.train()
    model.visualize_loss()