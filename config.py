import numpy as np
#参数文件，可以在这里配置相关参数
weight=np.array([0.6,-0.8]).reshape(2,1)         #用于生成线性回归数据集，预设权重
bias=2.3                                                            #用于生成线性回归数据集，预设偏置
low=-3                                                              #由均匀分布生成数据点，下界为-3
high=3                                                              #由均匀分布生成数据点，上届为3
num_sample=1000                                           #样本数