#目前只有随机梯度下降

class SGD_optimizer(): #比较简单，设个学习率而已
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate

    def opt(self,Model):
        for i in range(Model.layer_num):
            Model.peek().learning_rate=self.learning_rate
            Model.enqueue(Model.dequeue())
        return Model