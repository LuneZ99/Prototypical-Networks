import os
from tqdm import trange
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler

'''
transform = transforms.Compose([transforms.ToTensor()])
dataset_train = datasets.MNIST(root='/media/data/zhaoyin/', transform=transform)
dataset_test = datasets.MNIST(root='/media/data/zhaoyin/', train=False, transform=transform)

print(dataset_test.targets.shape[0])
sk = []
for way in trange(10, ncols=40):
    weights = [1 if label == way else 0 for label in dataset_test.targets]
    sampler = WeightedRandomSampler(weights, num_samples=50, replacement=False)
    sk.append(data.DataLoader(dataset_test, batch_size=5, sampler=sampler))

qk = []
for way in trange(10, ncols=40):
    weights = [1 if label == way else 0 for label in dataset_test.targets]
    sampler = WeightedRandomSampler(weights, num_samples=50, replacement=False)
    qk.append(data.DataLoader(dataset_test, batch_size=20, sampler=sampler))

for val, label in sk[0]:
    print(label.tolist())


from mnistm import MNISTM

mnist_root = '/media/data/zhaoyin'
root = '/media/data/zhaoyin/mnistm'
train = MNISTM(root=root, mnist_root=mnist_root)
test = MNISTM(root=root, mnist_root=mnist_root, train=False)
'''
while True:
    print('te')