import torch
from torch.autograd import Variable


#神经网络中的变量都是Variable
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
# requires_grad=True 允许误差反向传播
variable = Variable(tensor, requires_grad=True)
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)

print('t_out:', t_out, '\n', 'v_out:', v_out)
# 当前计算节点误差反向传递
v_out.backward()
# 打印反向传递的更新值（梯度）
# 在这里grad（梯度）=d(1/4×sum（variable×variable）)=1/2*variable
print("反向传递更新值", variable.grad)
# variable.data 返回variable中的tensor
print("variable中的tensor:", variable.data)
print("variable 转 numpy:", variable.data.numpy())
