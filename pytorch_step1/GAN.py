import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.functional as func
from torch.autograd import Variable

BATCH_SIZE = 64
N_IDEAS = 5
D_LR = 0.0001
G_LR = 0.0001
ART_COMPONENTS = 15

POINT_POINTS = torch.cat([torch.linspace(-1, 1, ART_COMPONENTS).view(1, -1) for _ in range(BATCH_SIZE)], 0)


def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * torch.pow(POINT_POINTS, 2).numpy() + (a - 1)
    paintings = torch.from_numpy(paintings).float()  # .type(torch.FloatTensor)
    return Variable(paintings)


# plt.plot(POINT_POINTS[0].numpy(),2*torch.pow(POINT_POINTS[0],2).numpy()+1)
# plt.plot(POINT_POINTS[0].numpy(),1*torch.pow(POINT_POINTS[0],2).numpy())
# plt.show()

D = torch.nn.Sequential(
    torch.nn.Linear(ART_COMPONENTS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)

G = torch.nn.Sequential(
    torch.nn.Linear(N_IDEAS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, ART_COMPONENTS),
)

D_optimzer = torch.optim.Adam(D.parameters(), lr=D_LR)
G_optimzer = torch.optim.Adam(G.parameters(), lr=G_LR)

for epoch in range(10000):
    artist_paints = artist_works()
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))
    G_paints = G(G_ideas)
    prob_artist0 = D(artist_paints)
    prob_artist1 = D(G_paints)
    loss_D = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    loss_G = torch.mean(torch.log(1. - prob_artist1))

    D_optimzer.zero_grad()
    loss_D.backward(retain_variables=True)
    D_optimzer.step()

    G_optimzer.zero_grad()
    loss_G.backward()
    G_optimzer.step()

    if epoch % 1000 == 0:
        plt.plot(POINT_POINTS[0].numpy(), 2 * torch.pow(POINT_POINTS[0], 2).numpy() + 1, lw=3, c='red',
                 label='Upper Bound')
        plt.plot(POINT_POINTS[0].numpy(), 1 * torch.pow(POINT_POINTS[0], 2).numpy(), lw=3, c='green',
                 label='Lower Bound')
        plt.plot(POINT_POINTS[0].numpy(), G_paints.data.numpy()[0], lw=2, c='yellow', label='Generated Paint')
        plt.text(-0.5, 2.3, 'D accuracy = %.2f' % prob_artist0.data.numpy().mean())
        plt.text(-0.5, 2, 'D score = %.2f' % -loss_D.data.numpy())
        plt.legend(loc='best')
        # plt.pause(0.5)
        plt.show()
