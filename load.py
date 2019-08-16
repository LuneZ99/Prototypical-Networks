import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as opt
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
TIME_LIMIT = 60 * 60 * 6
start = time.time()
num_classes = 10
feat_dim = 64
Sample_Num = 500
GPU_Num = 1
Batch_Sample_Num = 100


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
        ck = torch.zeros(1, 64).cuda(GPU_Num)
        for xj in sk:
            ck.add_(self.embedding(xj.view([1, 1, 28, 28])))
        ck = torch.div(ck, len(sk))
        return ck

    @staticmethod
    def distance(z0, z1):
        return torch.dist(z0, z1)

    def forward(self, ti):

        return self.embedding(ti)


net = FunctionJ().cuda(1)
net.load_state_dict(torch.load('net_params_500.pkl'))

transform = transforms.Compose([transforms.ToTensor()])
dataset_train = datasets.MNIST(root='/media/data/zhaoyin/', transform=transform)
dataset_test = datasets.MNIST(root='/media/data/zhaoyin/', train=False, transform=transform)

weights = [1] * Sample_Num
weights.extend([0] * (dataset_train.targets.shape[0] - Sample_Num))
sampler = WeightedRandomSampler(weights, num_samples=Sample_Num, replacement=False)
train = data.DataLoader(dataset_train, batch_size=1, sampler=sampler, pin_memory=True)

output = torch.Tensor(0, feat_dim).cuda(GPU_Num)
output_label = torch.Tensor(0).long().cuda(GPU_Num)
output_center = torch.Tensor(10, feat_dim).cuda(GPU_Num)

for data, label in tqdm(train):
    temp = net(data.cuda(GPU_Num))
    output = torch.cat((output, temp), 0)
    output_label = torch.cat((output_label, label.cuda(GPU_Num)), 0)
    output_center[int(label.item())].add_(temp.view(64))

for i in range(len(output_center)):
    output_center[i] = torch.div(output_center[i], Sample_Num)
print(output_center)

ouo = output.detach().cpu().numpy()
oul = output_label.detach().cpu().numpy()
ouc = output_center.detach().cpu().numpy()
np.save('64dim_origin.npy', ouo)
np.save('64dim_label.npy', oul)
np.save('64dim_center.npy', ouc)

