import os
import time
import tqdm
import torch
import numpy as np
import random as ra
import torch.nn as nn
import torch.optim as opt
import torchvision.datasets as data
from torch.nn import functional as fun
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


print('hello py_torch!')
N_C = 10 * 1
N_Q = 1
Learning_Rate = 0.02
Epoch_Number = 50000
TIME_LIMIT = 60 * 60 * 6
start = time.time()


class FunctionJ(nn.Module):
    def __init__(self):
        super(FunctionJ, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        print('Net init success!\n')

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def embedding(self, xi):
        xi = fun.max_pool2d(fun.relu(self.conv1(xi)), (2, 2))
        xi = fun.max_pool2d(fun.relu(self.conv2(xi)), (2, 2))
        xi = xi.view(-1, self.num_flat_features(xi))
        xi = fun.relu(self.fc1(xi))
        xi = fun.relu(self.fc2(xi))
        xi = self.fc3(xi)
        return xi

    def center(self, sk):
        ck = torch.zeros(1, 20).cuda()
        for xj in sk:
            ck.add_(self.embedding(xj))
        ck = torch.div(ck, len(sk))
        return ck

    @staticmethod
    def distance(z0, z1):
        return torch.dist(z0, z1)

    def forward(self, x):
        j = 0
        lck = torch.stack([self.center(x[0][k]) for k in range(N_C)])
        lqk = x[1]
        for k in range(N_C):
            j += (self.distance(self.embedding(lqk[k]), lck[k]) +
                  torch.logsumexp(
                      torch.stack(
                          [torch.neg(self.distance(self.embedding(lqk[k]), lck[ki]))
                           for ki in range(N_C) if ki != k]
                      ), 0) / (N_C * N_Q))
        return j


# dataset_train = data.MNIST(root='/media/data/zhaoyin/')
# dataset_test = data.MNIST(root='/media/data/zhaoyin/', train=False)

train = []
test = []
for i in range(10):
    train.append(np.load(os.path.join('dataset', 'MNIST', 'train_' + str(i) + '.npy')))
    test.append(np.load(os.path.join('dataset', 'MNIST', 'test_' + str(i) + '.npy')))


net = FunctionJ().cuda()
optimizer = opt.Adam(net.parameters(), lr=Learning_Rate, betas=(0.9, 0.99))
epoch = 0

while True:
    epoch += 1
# for epoch in range(Epoch_Number):

    inp = [[], []]

    for i in range(10):
        '''
        weights = [1 if 0 == label else 0 for data, label in dataset_train]
        sampler = WeightedRandomSampler(weights, num_samples=N_C+N_Q, replacement=False)
        inp.append(DataLoader(dataset_train, batch_size=3, sampler=sampler))
        '''
        li = [ra.randint(0, len(train[i])-1) for mi in range(10)]
        inp[0].append(torch.stack([torch.from_numpy(train[i][to]).view(1, 1, 28, 28)
                                   for to in li]).float().cuda())
        inp[1].append(torch.from_numpy(train[i][ra.randint(0, len(train[i])-1)]).view(1, 1, 28, 28).float().cuda())

    optimizer.zero_grad()
    loss = net(inp)
    loss.backward()
    optimizer.step()
    print('epoch=%s, loss=%s' % (epoch, loss.item()))

    if (time.time() - start) > TIME_LIMIT:
        break


