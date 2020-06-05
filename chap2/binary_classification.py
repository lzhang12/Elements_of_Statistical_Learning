"""
Binary Classification by Linear Regression and K-Nearest-Neighbor
in Chapter 2 of The Elements of Statistical Learning

author: zl
update: 20200604
"""

#%% import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)

#%% data generation
I = np.identity(2)
m_blue = np.random.multivariate_normal(mean=[1,0], cov=I, size=10)
m_orange = np.random.multivariate_normal(mean=[0,1], cov=I, size=10)

n_obs_per_class = 100
# X_blue = np.zeros((n_obs_per_class, 2))
# Y = np.zeros(n_obs_per_class)
X = []
Y = []
class_ = {0:m_blue, 1:m_orange}
for y, m in class_.items():
    for i in range(n_obs_per_class):
        mk = m[np.random.choice(len(m))]
        x = np.random.multivariate_normal(mean=mk, cov=1./5.*I, size=1)
        X.append(x)
        Y.append(y)

X = np.concatenate(X)
Y = np.array(Y)

#%% linear model
reg = LinearRegression().fit(X, Y)
beta = np.concatenate((reg.intercept_[np.newaxis], reg.coef_))

#%% kNN
k = 15
knn = KNeighborsClassifier(n_neighbors=k).fit(X, Y)

#%% plot
def plot_obs(ax, X, Y):
    X_blue, X_orange = X[Y==0], X[Y==1]
    ax.plot(X_blue[:,0], X_blue[:,1], 'o', markeredgecolor='blue', markerfacecolor='None', markersize=8)
    ax.plot(X_orange[:,0], X_orange[:,1], 'o', markeredgecolor='orange', markerfacecolor='None', markersize=8)


def plot_linear(ax, x1, x2, beta):
    # linear model decision boundary
    X1_lin = np.array([x1, x2])
    X2_lin = (0.5 - beta[1]*X1_lin - beta[0])/beta[2]
    ax.plot(X1_lin, X2_lin, 'k-')


def plot_decision_boundary(model, ax, xlim, ylim, N, cmap=ListedColormap(['blue','orange']), alpha=0.2, zorder=0):
    xmin, xmax = xlim
    ylim, ymax = ylim
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, N))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) >= 0.5
    ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=alpha, zorder=zorder)


fig, ax = plt.subplots(1, 2, figsize=(10,5))
xmin = -3
xmax = 4
ymin = -3
ymax = 4
xlim = [xmin, xmax]
ylim = [ymin, ymax]
N = 50

# linear model
plot_obs(ax[0], X, Y)
plot_linear(ax[0], xmin, xmax, beta)
plot_decision_boundary(reg, ax[0], xlim, ylim, N)
ax[0].set_title('Linear Regression')
ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)
ax[0].set_aspect('equal')

# knn model
plot_obs(ax[1], X, Y)
plot_decision_boundary(knn, ax[1], xlim, ylim, N)
ax[1].set_title('{}-KNN'.format(k))
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)
ax[1].set_aspect('equal')

plt.show()