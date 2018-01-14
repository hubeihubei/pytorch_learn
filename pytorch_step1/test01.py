import torch
import numpy as np


def func01():
    np_arr = np.arange(6).reshape((2, 3))
    torch_arr = torch.from_numpy(np_arr)
    tensor2array = torch_arr.numpy()
    print("np_arr:", np_arr)
    print('torch_arr:', torch_arr, type(torch_arr))
    print("tensor2array:", tensor2array)


def func02():
    data = [-1, -2, 4, 5]
    data2 = [[-1, -2, 4, 5]]
    # 将列表转为32bit张量
    tensor = torch.FloatTensor(data)
    # 将列表转为整数张量
    int_tensor = torch.IntTensor(data2)
    print(tensor, type(tensor))
    print(int_tensor, type(int_tensor))
    print("np 绝对值：", np.abs(data))
    print("torch 绝对值", torch.abs(tensor))


def func03():
    # 平均值
    arange = torch.arange(12)
    print('arange', arange, type(arange))
    print("average：", torch.mean(arange))
    # 三角函数：
    sin = torch.sin(arange)
    print('sin:', sin)


def func04():
    data = [[1, 2], [3, 4]]
    tensor = torch.FloatTensor(data)
    np_data = np.array(data)
    # 矩阵相乘
    nmm = np.matmul(np_data, np_data)
    tmm = torch.mm(tensor, tensor)

    print('nmm:',nmm,'\n','tmm:',tmm,'\n')

if __name__ == '__main__':
    # func01()
    # func02()
    # func03()
    func04()
