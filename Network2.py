import time
import torch
import torch.nn as nn
from tqdm import trange
import torch.optim as opt
from mnistm import MNISTM
import torch.utils.data as data
from torchvision import transforms
from torch.nn import functional as fun
import torchvision.datasets as datasets
from torch.utils.data.sampler import WeightedRandomSampler


print('hello py_torch!')
N_C = 10 * 1
N_Q = 1
Learning_Rate = 0.02
Epoch_Number = 50000
TIME_LIMIT = 60 * 60 * 1
start = time.time()
num_classes = 10
feat_dim = 64
Sample_Num = 500
GPU_Num = 5
Batch_Sample_Num = 20
Use_GPU = True


class FunctionJ(nn.Module):
    def __init__(self):
        super(FunctionJ, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, feat_dim)
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
        xi = self.fc2(xi)
        return xi

    def center(self, sk):
        if Use_GPU:
            ck = torch.zeros(1, 64).cuda(GPU_Num)
        else:
            ck = torch.zeros(1, 64)
        for xj in sk:
            ck.add_(self.embedding(xj.view([1, 1, 28, 28])))
        ck = torch.div(ck, len(sk))
        return ck

    @staticmethod
    def distance(z0, z1):
        return torch.dist(z0, z1)

    def forward(self, dataset_train, dataset_test):

        sk = []
        for way in range(10):
            weights = [1 if dataset_train.targets[i] == way else 0 for i in range(Sample_Num)]
            weights.extend([0] * (dataset_train.targets.shape[0] - Sample_Num))
            sampler = WeightedRandomSampler(weights, num_samples=Batch_Sample_Num, replacement=False)
            sk.append(data.DataLoader(dataset_train, batch_size=Batch_Sample_Num, sampler=sampler, pin_memory=True))

        qk = []
        for way in range(10):
            weights = [1 if dataset_train.targets[i] == way else 0 for i in range(Sample_Num)]
            weights.extend([0] * (dataset_train.targets.shape[0] - Sample_Num))
            sampler = WeightedRandomSampler(weights, num_samples=Batch_Sample_Num, replacement=False)
            qk.append(data.DataLoader(dataset_train, batch_size=1, sampler=sampler, pin_memory=True))

        vk = []
        for way in range(10):
            weights = [1 if label == way else 0 for label in dataset_test.targets]
            sampler = WeightedRandomSampler(weights, num_samples=100, replacement=False)
            vk.append(data.DataLoader(dataset_test, batch_size=1, sampler=sampler, pin_memory=True))

        # vk = data.DataLoader(dataset_test, batch_size=10000, pin_memory=True)
        # print(sk, qk, vk)
        ck = []
        for num in sk:
            for datas, _ in num:
                if Use_GPU:
                    datas = datas.cuda(GPU_Num)
                ck.append(self.center(datas))

        j = 0
        for i in qk:
            for datas, label in i:
                if Use_GPU:
                    datas = datas.cuda(GPU_Num)
                em = self.embedding(datas)
                j += (self.distance(em, ck[label]) + torch.logsumexp(
                      torch.stack([torch.neg(self.distance(em, ci)) for ci in ck]), 0))
        j /= 200

        acc = 0.0

        for i in vk:
            for datas, label in i:
                if Use_GPU:
                    datas = datas.cuda(GPU_Num)
                em = self.embedding(datas)
                ls = [self.distance(em, ci) for ci in ck]
                vl = ls.index(min(ls))
                # print(label.item(), vl)
                if vl == label.item():
                    acc += 1
        acc /= 1000.0

        '''
        for i, label in vk:
            for datas in i:
                em = self.embedding(datas.cuda(GPU_Num).view([1, 1, 28, 28]))
                ls = [self.distance(em, ci) for ci in ck]
                vl = ls.index(min(ls))
                # print(label.item(), vl)
                if vl == label.item():
                    acc += 1
        acc /= 10000.0
        '''
        return j, acc


net = FunctionJ()
if Use_GPU:
    net = net.cuda(GPU_Num)
optimizer = opt.Adam(net.parameters(), lr=Learning_Rate, betas=(0.9, 0.99))
epoch = 0
transform = transforms.Compose([transforms.ToTensor()])
# dataset_train = datasets.MNIST(root='/media/data/zhaoyin/', transform=transform)
# dataset_test = datasets.MNIST(root='/media/data/zhaoyin/', train=False, transform=transform)
mnist_root = '/media/data/zhaoyin'
root = '/media/data/zhaoyin/mnistm'
dataset_train = MNISTM(root=root, mnist_root=mnist_root, transform=transform)
dataset_test = MNISTM(root=root, mnist_root=mnist_root, train=False, transform=transform)
loss_ls = [0] * 10
acc_ls = [0] * 10
# print(loss_ls, acc_ls)

while True:
    p = epoch % 10
    epoch += 1
    optimizer.zero_grad()
    loss, acc = net(dataset_train, dataset_test)
    loss_ls[p], acc_ls[p] = loss.item(), acc
    loss.backward()
    optimizer.step()

    print('epoch=%s, loss=%s, acc=' % (epoch, loss.item()), acc)

    if (time.time() - start) > TIME_LIMIT or sum(loss_ls) < 0.01:
        with open('N2.txt', 'a') as f:
            f.write('sample_num = %s, epoch = %s, loss = %s, acc = %s \n'
                    % (Sample_Num, epoch, sum(loss_ls)/10, sum(acc_ls)/10))
        torch.save(net.state_dict(), 'net_params_500.pkl')
        print('超时 or 收敛，训练停止')
        break
