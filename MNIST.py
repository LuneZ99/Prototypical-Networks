import os
import torch
import numpy as np
import torchvision.datasets as data


train = data.MNIST('/media/data/zhaoyin/')
test = data.MNIST(root='/media/data/zhaoyin/', train=False)


print('Hello MNIST')
'''
print(type(np.array([])))

dic = {}

for t in test:
    if t[1] in dic.keys():
        dic[t[1]].append(np.array(t[0]))
    else:
        dic[t[1]] = [np.array(t[0])]

for key, value in dic.items():
    np.save(os.path.join('dataset', 'MNIST', 'test_'+str(key)+'.npy'), np.array(value))


for i in range(10):
    tr = np.load(os.path.join('dataset', 'MNIST', 'train_' + str(i) + '.npy'))
    te = np.load(os.path.join('dataset', 'MNIST', 'test_' + str(i) + '.npy'))
    tr.astype(np.float32)
    te.astype(np.float32)
    np.save(os.path.join('dataset', 'MNIST', 'train_' + str(i) + '.npy'), tr)
    np.save(os.path.join('dataset', 'MNIST', 'test_' + str(i) + '.npy'), te)
'''


tr = np.load(os.path.join('dataset', 'MNIST', 'train_' + str(0) + '.npy'))
x = torch.from_numpy(tr[0])
print(x.view(1,1,28,28).shape)

print(1)
