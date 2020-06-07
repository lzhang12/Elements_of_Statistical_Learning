"""
Binary Classification by Linear Regression and K-Nearest-Neighbor
in Chapter 2 of The Elements of Statistical Learning

author: zl
update: 20200606
"""

#%% import
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from importlib import resources

np.random.seed(0)
plt.style.use('../utils/default_plot_style.mplstyle')

SAVE_FIGURE = False
DPI = 300
DIR_FIGURE = '../figure'
CHAPTER = 'ch2'
PROBLEM = 'binary_class'

#%% data generation
def sample_generator(N, mean=None, n_mean=10, var=0.2):
    """
    Generate binary (blue=0, orange=1) samples
        - equal number of sample each class
        - n_mean points is first generated for each class by Gaussian distribution
        - N points is generated using Gaussian distribution by randomly picking a mean in n_mean, and a variance reduced var_reduce

    Argument:
        N = number of sample per class
        n_mean = number of means

    Return:
        X = sample data points, shape = (2*N, 2)
        Y = class labels, 0=blue, 1=orange, shape = (2*N, )
        mean = mean values of the points to generate sample data
    """

    I = np.identity(2)
    if mean is None:
        m_blue = np.random.multivariate_normal(mean=[1,0], cov=I, size=n_mean)
        m_orange = np.random.multivariate_normal(mean=[0,1], cov=I, size=n_mean)
        mean = (m_blue, m_orange)

    m_blue, m_orange = mean
    X = []
    Y = []
    class_ = {0:m_blue, 1:m_orange}
    for y, m in class_.items():
        for _ in range(N):
            mk = m[np.random.choice(n_mean)]
            x = np.random.multivariate_normal(mean=mk, cov=var*I, size=1)
            X.append(x)
            Y.append(y)

    X = np.concatenate(X)
    Y = np.array(Y)

    return X, Y, mean

def plot_obs(X, Y, xlim=None, ylim=None, ax=None, is_save=False, dpi=300):
    """
    plot sample observations

    Argument:
        X = samples of dimension (N, 2)
        Y = labels of dimension (N, )

    Return:
        fig = figure
        ax = axis
    """
    X_blue, X_orange = X[Y==0], X[Y==1]

    if ax is None: fig, ax = plt.subplots(1, 1)
    ax.plot(X_blue[:,0], X_blue[:,1], 'o', markeredgecolor='blue', markerfacecolor='None', markersize=8)
    ax.plot(X_orange[:,0], X_orange[:,1], 'o', markeredgecolor='orange', markerfacecolor='None', markersize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    if is_save == True:
        fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM, 'samples']))
        plt.savefig(fn, dpi=dpi)

    return fig, ax

N = 100
N_MEAN = 10
VAR = 0.2
X, Y, mean = sample_generator(N, n_mean=N_MEAN, var=VAR)
fig, ax = plot_obs(X, Y, xlim=[-3, 4], ylim=[-3, 4])

#%% Binary classification model
def plot_decision_boundary(model, X, Y, title=None, linear=False, xlim=None, ylim=None, n_point=50, cmap=ListedColormap(['blue','orange']), alpha=0.2, zorder=0, is_save=False):
    """
    Plot decision boundary of model prediction
    """
    fig, ax = plot_obs(X, Y, xlim, ylim)

    xmin, xmax = ax.xlim() if xlim is None else xlim
    ymin, ymax = ax.ylim() if ylim is None else ylim

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_point), np.linspace(ymin, ymax, n_point))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) >= 0.5
    ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=alpha, zorder=zorder)
    ax.set_title(title)

    if linear is True:
        beta = np.concatenate((model.intercept_[np.newaxis], model.coef_))
        X1_lin = np.array([xmin, xmax])
        X2_lin = (0.5 - beta[1]*X1_lin - beta[0])/beta[2]
        ax.plot(X1_lin, X2_lin, 'k-')

    if is_save == True:
        fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM, title]))
        plt.savefig(fn, dpi=DPI)

    return fig

# linear model
reg = LinearRegression().fit(X, Y)
title = 'Linear Regression'
plot_decision_boundary(reg, X, Y, linear=True, xlim=[-3, 4], ylim=[-3, 4], title=title, is_save=SAVE_FIGURE)

# kNN
k = 15
knn = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
title = '{}-Nearest Neighbor Classifier'.format(k)
plot_decision_boundary(knn, X, Y, xlim=[-3, 4], ylim=[-3, 4],title=title, is_save=SAVE_FIGURE)

k = 1
knn = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
title = '{}-Nearest Neighbor Classifier'.format(k)
plot_decision_boundary(knn, X, Y, xlim=[-3, 4], ylim=[-3, 4], title=title, is_save=SAVE_FIGURE)
plt.show()

# Bayesian
class BayesianClassifier(object):
    """
    Bayesian classifier using the data generator
    """
    def __init__(self, n_class, mean, var):
        self.n_class = n_class
        self.mean = mean
        self.cov = var*np.identity(n_class)

    def Pr(self, X):
        """
        Compute probability of each class
        """
        p = np.zeros((X.shape[0], self.n_class))
        for i in range(self.n_class):
            for m in self.mean[i]:
                p[:,i] += multivariate_normal.pdf(X, mean=m, cov=self.cov)
        return p

    def predict(self, X):
        """
        Make prediction
        """
        Y = np.argmax(self.Pr(X), axis=1)
        return Y

bayes = BayesianClassifier(2, mean, VAR)
title = 'Bayesian Classifier'
plot_decision_boundary(bayes, X, Y, xlim=[-3, 4], ylim=[-3, 4], title=title, is_save=SAVE_FIGURE)
plt.show()

#%% Test Error vs k
def plot_knn_err(kk, train_err, test_err, lin_train_err=None, lin_test_err=None, bayes_err=None, is_save=False):
    """
    Plot test and training error for different degrees of freedoms and number of nearest neighbors
    """

    fig, ax = plt.subplots(1, 1)

    ax.plot(kk, train_err, 'o-', label='train', markersize=4)
    ax.plot(kk, test_err, 'o-', label='test', markersize=4)

    ax.set_xscale('log')
    ax.set_xticks(kk[::2])
    ax.set_xticklabels(kk[::2])
    ax.set_xlabel(r'$k$ - numebr of nearest neighbors')
    ax.set_ylabel('Error')
    ax.minorticks_off()

    dof = (2*N/kk).astype(np.int)
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(kk[::2])
    ax2.set_xticklabels(dof[::2])
    ax2.set_xlabel(r'degree of freedoms $N/k$')
    ax2.minorticks_off()

    if lin_train_err:
        xlin = 70
        ax.plot(xlin, lin_train_err, 's', color='C0', markersize=8)
        ax.plot(xlin, lin_test_err, 's', color='C1', markersize=8)
        ax.text(xlin, lin_test_err+0.01, 'linear', ha='center', va='bottom')

    if bayes_err:
        xlim = ax.get_xlim()
        ax.plot(xlim, [bayes_err]*2, 'k--', label='Bayes')

    ax.legend(loc='best')

    if is_save == True:
        fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM, 'knn_err']))
        plt.savefig(fn, dpi=DPI)

    return fig, ax


X_test, Y_test, _ = sample_generator(5000, mean)

kk = np.unique(np.logspace(0, np.log10(151), num=20).astype(np.int))
knn_train_err = []
knn_test_err = []
for k in kk:
    cls = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
    acc = accuracy_score(Y, cls.predict(X))
    knn_train_err.append(1-acc)
    acc = accuracy_score(Y_test, cls.predict(X_test))
    knn_test_err.append(1-acc)

lin_train_err = 1 - accuracy_score(Y, reg.predict(X)>0.5)
lin_test_err = 1 - accuracy_score(Y_test, reg.predict(X_test)>0.5)

bayes_err = 1 - accuracy_score(Y_test, bayes.predict(X_test))

plot_knn_err(kk, knn_train_err, knn_test_err, lin_train_err, lin_test_err, bayes_err, is_save=True)
plt.show()
