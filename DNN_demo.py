import numpy as np
import matplotlib.pyplot as plt
from Loss import mse_loss
from optimizers import SGD_optimizer
from Model import Model, Dense
from dataloader import randDataGenerator, dataset_generator
from config import *
import sys
import argparse

#这里主要是对深度神经网络测试，这里用的自己生成的线性回归数据集，根据loss曲线，可以判断通过测试
#当然，也可以使用自己的数据集进行训练
#目前该项目也支持logistic激活函数，如果你有可以使用深度神经网络的分类数据集，也可以使用该项目
class Linear_regression():
    def __init__(self,dataset_generator,
                epoch=20,
                batch_size=20,
                optimizer=SGD_optimizer(learning_rate=0.01),
                loss="mse"):

        self.model=Model([Dense(dataset_generator.shape[1]-1,3,activation="relu"),
                         Dense(3,1)])   #这里做个小测试，只是为了测试深度网络能否收敛（能不能反向传播）
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
    parser = argparse.ArgumentParser('Deep Neural Network')
    parser.add_argument('--epoch', type=int, default=20,
                        help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="batch_size必须能整除样本数，样本数默认1000，可在config文件中设置")
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help="随机梯度下降学习率设置")
    args = parser.parse_args()
    print(args)
    stdout_backup = sys.stdout
    log_file = open("message_DNN.log", "w")
    sys.stdout = log_file
    data=randDataGenerator(num_sample=num_sample,low=low,high=high)
    dataset=dataset_generator(data,weight=weight,bias=bias,no_zenguang=True) #其中有一个维度为增广维度
    model=Linear_regression(dataset, epoch=args.epoch, batch_size=args.batch_size,
                            optimizer=SGD_optimizer(learning_rate=args.learning_rate))
    model.train()
    log_file.close()
    sys.stdout = stdout_backup
    model.visualize_loss()