import numpy as np

#根据实验手册要求，默认1000个样本，特征数为2，均匀分布范围-3到3
#线性回归的数据集生成器，后面会手撕一个用于logistic回归数据集生成器

def randDataGenerator(num_sample=1000,dim=2,low=-3,high=3):
    """
    num_sample:样本个数
    dim:特征向量维度
    low:均匀分布下界
    high：均匀分布上界
    """
    data=np.random.uniform(low,high,size=(dim,num_sample)) #所有特征向量默认列向量
    _one=np.ones((1,num_sample))
    data=np.concatenate((data,_one),axis=0)
    return data
#测试一下，保证类型，大小符合预期
#print(randDataGenerator().dtype) 
#print(randDataGenerator().shape)
#print(all(randDataGenerator()[2,:]==1)) #确保增广有效

def dataset_generator(data,weight=None,bias=None, gauss_noise=True,mu=0.,sigma=0.01,only_label=False,no_zenguang=False):
    """
    weight：一个列向量，设置线性回归的权重
    bias：一个float，设置偏置
    gauss_noise：是否设置高斯噪声
    mu:高斯噪声均值，gauss_noise为true时生效
    sigma:高斯噪声方差，gauss_noise为true时生效
    only_label:是否只返回标签，若为False，那么将会返回样例（特征向量+标签）的表
    no_zenguang:返回无增广数据集
    """
    assert weight.shape[1]==1 #确保列向量
    #bias输入为数值，转换为矩阵
    bias=np.array([bias]).reshape(1,1)
    W=np.concatenate((weight,bias),axis=0)
    row,column=data.shape #column即为样本数量
    assert W.shape[0]==row
    label=np.matmul(data.T,W)
    label=label+sigma*(np.random.standard_normal(size=(column,1))+mu)
    if only_label:
        return label
    else:
        if no_zenguang:
            return np.delete(np.concatenate((data.T,label),axis=1),2,axis=1)
        else:
            return np.concatenate((data.T,label),axis=1)