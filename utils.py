#用来放数据结构的，用到什么写什么，这里只用了队列
class Queue:
    def __init__(self):
        self.item=[]
    def enqueue(self,item):
        self.item.insert(0,item) 
    def dequeue(self):
        return self.item.pop()
    def isEmpty(self):
        return self.item==[]
    def size(self):
        return len(self.item)
    def peek(self): #栈的方法，这里需要用上
        return self.item[self.size()-1]