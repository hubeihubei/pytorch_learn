import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

step = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
x_np = np.sin(step)
y_np = np.cos(step)
# plt.plot(step, x_np, 'r-', label='input(sin)')
# plt.plot(step, y_np, "b-", label='target(cos)')
# plt.legend(loc='best')
# plt.show()


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            num_layers=1,
            hidden_size=32,
            batch_first=True
        )

        self.out = torch.nn.Linear(32, 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, None)
        #输出每一次Time_step 的预测值
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn=RNN()
optimzer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=torch.nn.MSELoss()
plt.figure(1,figsize=(12,5))
plt.ion()
for step in range(100):
    #pytorch 支持动态神经网络
    dynamic_steps=np.random.randint(1,4)
    start,end=np.pi*step,np.pi*(step+dynamic_steps)
    steps=np.linspace(start,end,TIME_STEP*dynamic_steps,dtype=np.float32)
    # print(len(steps))
    x_np=np.sin(steps)
    y_np=np.cos(steps)
    print("x_np:",x_np.shape)
    x=Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis]))
    y=Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))
    prediction,h_state=rnn(x)
    # print(h_state)
    loss=loss_func(prediction,y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')

    # plt.draw()
    # plt.pause(0.05)
plt.show()