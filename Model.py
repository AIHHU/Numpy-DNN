import numpy as np
from utils import Queue
#编写Dense类，包含，参数初始化，以及前向推理函数 数据包括反向传播误差项
#目前支持批量前向计算，但不支持批量反向传播，反向传播基于列表
#个人感觉：有点bad smell，但是貌似没有重构的必要
#目前只有Dense，CNN后续有时间再开发
#激活函数目前只支持logistic和relu，或者没有（线性）
class Dense(): 
    def __init__(self,in_dim,out_dim,activation=None):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.activation=activation
        assert self.activation=="logistic" or self.activation=="relu" or self.activation==None #目前只支持两种损失函数
        self.Weight, self.bias=self.initialize()
        self.res=[] #并非完全是误差项，是误差项与参数W的乘积
        self.Input=None
        self.output=None
        self.learning_rate=None #由优化器函数进行更改即可
        self.output_layer=False #初始化时是False，由model类封装时最后一层自动改为True
    
    def initialize(self): #使用glorot初始化，偏置用0初始化
        if self.activation=="logistic":
            sigma=np.sqrt(32/(self.in_dim+self.out_dim))
        else:
            sigma=np.sqrt(2/(self.in_dim+self.out_dim))
        return sigma*(np.random.standard_normal(size=(self.out_dim,self.in_dim))), np.zeros((self.out_dim,1))

    def forward(self,x): #定义前向计算方法，默认x形状为(特征数,batch_size)
        assert x.shape[0]==self.in_dim
        self.Input=x
        z=np.matmul(self.Weight,x)+self.bias
        if self.activation=="logistic":
            self.output=1/(1+np.exp(-z))
            return self.output
        elif self.activation=="relu":
            self.output=np.maximum(z,0.)
            return self.output
        else:
            self.output=z
            return self.output #线性回归专用，狗头保命
    
    def backward(self,res_last=None): #反向传播更新
        """
        res_last:列表|矩阵，输出层为输入损失矩阵，隐藏层为误差项与参数乘积，为列表
        """
        self.res=[] #每反向传播一次，一定要清空
        dw=np.zeros(self.Weight.shape) #更新梯度用的中间变量
        db=np.zeros(self.bias.shape) #更新偏置用的中间变量

        if self.output_layer==True:
            """
            此时res_last是一个二维矩阵,为列向量
            输出是一个行向量（1，batch_size）
            """
            assert self.activation==None or self.activation=="logistic" #不允许用relu做输出层

            if self.activation=="logistic": #输出层
                res_last=((self.output*(1-self.output)).T * res_last).tolist()
            else:
                res_last=res_last.tolist()
            #这里直接更新参数,并保存误差项与参数乘积
            for idx,item in enumerate(res_last):
                item=np.array([item])
                dw += np.dot(item,self.Input[:,idx].reshape((1,-1)))
                db += item
                self.res.append(np.dot(self.Weight.T,item))

            self.Weight=self.Weight-self.learning_rate*dw/len(self.res)
            self.bias=self.bias-self.learning_rate*db/len(self.res)
            
        else:
            """
            如果不是输出层，res_last为列表，其中元素为列向量，计算该层误差项
            """
            assert self.activation==None or self.activation=="relu" #不允许用logistic做隐藏层
            #这里做了一些优化，一边压入列表，一边计算梯度
            #但是有重复代码，有点bad smell
            if self.activation==None:
                for idx,item in enumerate(res_last):
                    
                    item=np.dot(np.eye(self.out_dim),item) #先计算该层误差项,
                    dw += np.dot(item,self.Input[:,idx].reshape((1,-1)))
                    db += item
                    self.res.append(np.dot(self.Weight.T,item))
            else:
                for idx,item in enumerate(res_last):
                    temp_diag=(self.output[:,idx]>0).astype("float64")
                    temp_diag=temp_diag.reshape(self.out_dim)

                    item=np.dot(np.diag(temp_diag),item) #先计算该层误差项

                    dw += np.dot(item,self.Input[:,idx].reshape((1,-1)))
                    db += item
                    self.res.append(np.dot(self.Weight.T,item))
            self.Weight=self.Weight-self.learning_rate*dw/len(self.res)
            self.bias=self.bias-self.learning_rate*db/len(self.res)
            
class Model(Queue): #基于队列结构实现model类，对model进行封装
    def __init__(self, layer_list):
        super(Model, self).__init__()
        for item in layer_list:
            self.item.append(item) #这里先追加，初始化层队列
        self.layer_num=self.size()
        self.peek().output_layer=True #将最后一层自动设置为输出层

    def forward(self,x):
        for item in self.item:
            x=item.forward(x)
        return x
    
    def backpropagate(self, original_res):
        #temp_for_res=original_res 后来想想，多此一举
        for i in range(self.layer_num):
            self.peek().backward(res_last=original_res)
            original_res=self.peek().res
            #接下来，尾到头上
            self.enqueue(self.dequeue())