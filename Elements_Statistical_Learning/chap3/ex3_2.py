"""
Simulation for Exercise 3.2
"""

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from tabulate import tabulate

plt.style.use('../utils/default_plot_style.mplstyle')
np.set_printoptions(precision=3)

SAVE_FIGURE = True
DPI = 300
DIR_FIGURE = '../figure'
CHAPTER = 'ch3'
PROBLEM = 'Ex_32'

#%% generate random sample
np.random.seed(42)

# set params
beta = np.array([1, 0.4, 0.5, 0.2])  # low --> high order
sigma = 1
N = 100

x = (np.random.rand(N) - 0.5)*5
# x = np.random.normal(0, 1, N)
epsilon = np.random.normal(0, sigma, N)
y = np.polyval(np.flip(beta), x) + epsilon

#%% linear regression
poly_reg = lambda x: np.stack([x**0, x**1, x**2, x**3]).T
X = poly_reg(x)
betah = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print('Given beta: {}'.format(beta))

yh = np.polyval(np.flip(betah), x)
sigmah_sq = np.sum((y-yh)**2)/(N-4)
Var_betah = (np.linalg.inv(np.dot(X.T, X))*sigmah_sq).diagonal()
z = betah/np.sqrt(Var_betah)

tab = np.stack([betah, np.sqrt(Var_betah), z]).T
print(tabulate(tab, headers=['Coef.', 'Std. Err.', 'Z Score']))

#%% confidence interval
# 1. point-wise
l = norm.ppf(0.975) # 95% quantile
Np = 100
xp = np.linspace(np.min(x), np.max(x), Np)
Xp = poly_reg(xp)
yp = np.polyval(np.flip(betah), xp)
interval = np.sqrt(sigmah_sq*np.dot(np.dot(Xp, np.linalg.inv(np.dot(X.T, X))), Xp.T).diagonal())
band_1 = [yp - l*interval, yp + l*interval]

# 2. confidence set
l2 = chi2.ppf(0.95, 4)
r = np.sqrt(sigmah_sq*l2)

# sample on a 4-d sphere surface
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
def gamma_sample(n, r):
    """
    sample n times of gamma
    """
    gamma = []
    for _ in range(n):
        phi1, phi2 = np.random.rand(2)*np.pi
        phi3 = np.random.rand(1)*np.pi*2
        g = r*np.array([np.cos(phi1),
                        np.sin(phi1)*np.cos(phi2),
                        np.sin(phi1)*np.sin(phi2)*np.cos(phi3),
                        np.sin(phi1)*np.sin(phi2)*np.sin(phi3)])
        gamma.append(g)
    return gamma

Nb = 10
gamma = gamma_sample(Nb, r)
y_band2 = np.zeros((Np, Nb))
# beta_band2 = []
for i, g in enumerate(gamma):
    W, V = np.linalg.eig(np.dot(X.T, X))
    D = np.diag(np.sqrt(W))
    _beta = betah - np.dot(np.dot(V, np.linalg.inv(D)), g)
    # beta_band2.append(_beta)
    y_band2[:,i] = np.polyval(np.flip(_beta), xp)

#%% plot
fig, ax = plt.subplots(1,1,figsize=(5,5))
h1 = ax.scatter(x, y, s=20, color='C0', label='data')
h2 = ax.plot(xp, yp, lw=2, color='C1', zorder=5, label='Fitted')
h3 = ax.fill_between(xp, band_1[0], band_1[1], alpha=0.3, color='C1',zorder=4, label='Band (pointwise)')
h4 = ax.plot(xp, y_band2, lw=1, color='C2',zorder=2, label='Band (set)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(handles=(h1, h2[0], h3, h4[0]), loc='best')
plt.show()

if SAVE_FIGURE == True:
    fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM]))
    plt.savefig(fn, dpi=DPI)