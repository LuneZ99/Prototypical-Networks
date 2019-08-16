from sklearn import decomposition as de
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.load('64dim_origin.npy')
label = np.load('64dim_label.npy')
'''
x += 155
lda = de.LatentDirichletAllocation(n_topics=10)
y = lda.fit_transform(x)
print(y)
'''
pca = de.PCA(n_components=3)
y = pca.fit_transform(x)

color = ['aqua', 'black', 'blue', 'gold', 'green', 'orange', 'pink', 'purple', 'red', 'lightgreen']

xl = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
      np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
yl = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
      np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
zl = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
      np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]

for i in range(500):
    xl[label[i]] = np.append(xl[label[i]], [y[..., 0][i]], axis=0)
    yl[label[i]] = np.append(yl[label[i]], [y[..., 1][i]], axis=0)
    zl[label[i]] = np.append(zl[label[i]], [y[..., 2][i]], axis=0)

ax = plt.subplot(111, projection='3d')
for way in range(10):
    ax.scatter(xl[way], yl[way], zl[way], c=color[way], alpha=.6)


ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.savefig("test.png")
