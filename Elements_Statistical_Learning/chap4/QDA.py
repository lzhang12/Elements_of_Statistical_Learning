"""
Classification of Vowel Data using Quadratic Discriminant Analysis

For visualization, we first reduce the dimension to 2 canonical variates, and then perrform QDA for the transformed data.
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import palettable
from matplotlib.colors import ListedColormap

plt.style.use('../utils/default_plot_style.mplstyle')
np.set_printoptions(precision=3)

SAVE_FIGURE = False
DPI = 300
DIR_FIGURE = '../figure'
CHAPTER = 'ch4'
PROBLEM = 'vowel_data_QDA'

#%% Vowel data
train = pd.read_csv('../data/vowel_train.csv')
X = train[[i for i in train.columns if i.startswith('x')]].values
y = train.y.values

X = X - np.mean(X, axis=0)

#%% Find discriminant coordinates
classes, counts = np.unique(y, return_counts=True)
N = X.shape[0]
K = len(classes)
p = X.shape[-1]

# K*p matrix of class centroids
M = np.zeros((K,p))
for i, k in enumerate(classes):
    M[i,:] = np.mean(X[y==k,:], axis=0)

# Within-class covariance
# method 1
W = np.zeros((p, p))
for i, k in enumerate(classes):
    Xk = X[y==k, :] - M[i,:]
    W += np.dot(Xk.T, Xk)

# method 2
# Xc = np.array([M[k-1,:] for k in y])
# W = np.dot((X-Xc).T, X-Xc)

# Between-class covariance
B = np.dot(M.T, np.dot(np.diag(counts), M))

# Test B + W = T
T = np.dot(X.T, X)
res = B + W - T
# print(res)

# discriminant variables
D, V = np.linalg.eig(np.dot(np.linalg.inv(W), B))
V = V[:, np.argsort(D)[::-1]]

# V = np.dot(V, np.diag(np.sqrt(D)))
Z = np.dot(X, V)
Zm = np.dot(M, V)

#%% # Decision boundary on the first two canonical variates
coords = [0,1]

# common variance
Zr = Z[:, coords]

# generate mesh
xmin, xmax = (-2, 2)
ymin, ymax = (-2, 1.5)
n_point = 500
xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_point), np.linspace(ymin, ymax, n_point))
X_mesh = np.c_[xx.ravel(), yy.ravel()]

# quadratic discriminant function
delta = np.zeros((n_point*n_point, K))
for i, k in enumerate(classes):
    Zk = Zr[y==k,:]
    muk = np.mean(Zk, axis=0)
    sigmak = np.dot((Zk-muk).T, Zk-muk)/(counts[i])
    det = np.linalg.det(sigmak)
    inv = np.linalg.inv(sigmak)
    delta[:,i] = -1/2.*np.log(det) - 1/2.*np.sum((X_mesh-muk)*np.dot(inv, (X_mesh-muk).T).T, axis=1)

delta = delta.reshape((n_point, n_point, K))
delta = np.argmax(delta, axis=-1)

# plot
cm = palettable.cartocolors.qualitative.Bold_10.mpl_colormap
c_sample = [cm(i/11) for i in y]
c_class = [cm(i/11) for i in classes]
cmap = ListedColormap(c_class)

fig, ax = plt.subplots(1, 1,figsize=(7,7))
ax.scatter(Z[:,coords[0]], Z[:,coords[1]], s=20, facecolors='None', edgecolors=c_sample, zorder=3)
ax.scatter(Zm[:,coords[0]], Zm[:,coords[1]], s=200, facecolors=c_class, zorder=2)
ax.contour(xx, yy, delta, colors='black', levels=classes-1, antialiased=True, alpha=1, zorder=1)
# ax.pcolormesh(xx, yy, delta, cmap=cmap, alpha=0.1, zorder=1) # could be slow
ax.set_xlabel('canonical coordinate '+str(coords[0]))
ax.set_ylabel('canonical coordinate '+str(coords[1]))

if SAVE_FIGURE == True:
    fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM]))
    plt.savefig(fn, dpi=DPI)
else:
    plt.show()

# %%