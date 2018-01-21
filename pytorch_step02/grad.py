import torch
from torch.autograd import Variable


def grad01():
    m=Variable(torch.FloatTensor([[2,3]]),requires_grad=True)
    print(m)
    n=Variable(torch.zeros(1,2))
    n[0,0]=m[0,0]**2
    n[0,1]=m[0,1]**3
    print(n)
    # n.backward(m.data,retain_graph=True)
    # print(m.grad.data)
    # m.grad.data.zero_()
    # 因为n.backward(m.data)求出来的是得到的梯度乘以对应的元素（4*2）（27*3），所以传入元素为一张量求的真是结果
    n.backward(torch.FloatTensor([[1,1]]))
    print(m.grad.data)

def grad02():
    m=Variable(torch.FloatTensor([[2,3]]),requires_grad=True)
    j= torch.zeros(2,2)
    k=Variable(torch.zeros(1,2))
    # m.grad.data.zero_()
    k[0,0]=m[0,0]**2+m[0,1]*3# 对m[0,0]求导=4，对m[0,1]求导=3
    k[0,1]=m[0,1]**2+m[0,0]*2# 对m[0,0]求导=2，对m[0,1]求导=6
    # 如果k每个元素是由多个变量组成，k的每个元素对m求导，k每个元素求导结果为对每个元素中变量求导的和（4+2,3+6）
    k.backward(torch.FloatTensor([[1,1]]),retain_graph=True)
    print(m.grad.data)
    m.grad.data.zero_()
    # retain_graph=True 默认为False，表示反向传播后，这个计算图的内存会被释放掉，就没办法进行第二次反向传播了，所以要设置为True
    k.backward(torch.FloatTensor([[1,0]]),retain_graph=True)
    j[:,0]=m.grad.data
    # 对m的计算图清零
    m.grad.data.zero_()
    k.backward(torch.FloatTensor([[0,1]]))
    j[:,1]=m.grad.data
    print(j)


if __name__ == '__main__':
    grad02()
