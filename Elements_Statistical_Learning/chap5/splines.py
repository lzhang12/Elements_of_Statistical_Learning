"""
Example of Piecewise Polynomial and Spline
Fig 5.2 & 5.9

Comment:
- In Figure 5.9, compare my implementation with the csaps implementation: https://csaps.readthedocs.io/en/latest/api.html
- Python does not have function for computing smoothing splines with given degree of freedoms like R.
"""

#%%
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
from utils import utils
import matplotlib.pyplot as plt

plt.style.use('../utils/default_plot_style.mplstyle')
np.set_printoptions(precision=3)

# %% Figure 5.2 (the real function is different)
SEED = 200
np.random.seed(SEED)

# generate artificial data
N = 50
var = 0.3
M = 4 #
xi = [0.3, 0.7]
X = np.linspace(0,1,1000)
f = lambda x: np.cos(x*np.pi*2)
Y = f(X)
Xd = np.sort(np.random.random(N))
Yd = f(Xd) + np.random.normal(scale=var, size=N)

# spline fit
spline_base = lambda x, xi, k: (x>xi)*(x-xi)**k
spline = lambda x, xi: np.concatenate((np.column_stack([x**i for i in range(M)]), np.column_stack([spline_base(x, xi_, M-1) for xi_ in xi])), axis=1)

H = spline(Xd, xi)
beta = np.linalg.inv(H.T@H)@H.T@Yd  # linear regression coefficient
Xfit = np.linspace(0,1,200)
Hfit = spline(Xfit, xi)
Yfit = np.dot(Hfit, beta)

# natural spline
xi_bnd = [0, *xi, 1]
df = lambda x, xi, xiK: ((x-xi)*(x-xi)**3 - (x-xiK)*(x-xiK)**3)/(xiK - xi)
natural_spline_base = lambda x, xi: [df(x, xi_, xi[-1]) - df(x, xi[-2], xi[-1]) for xi_ in xi[0:-2]]
natural_spline = lambda x, xi: np.concatenate((np.column_stack([x**i for i in range(2)]), np.column_stack(natural_spline_base(x, xi_bnd))), axis=1)

H_nat = natural_spline(Xd, xi)
beta_nat = np.linalg.inv(H_nat.T@H_nat)@H_nat.T@Yd  # linear regression coefficient
Hfit_nat = natural_spline(Xfit, xi)
Yfit_nat = np.dot(Hfit_nat, beta_nat)

# plot
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(X, Y, color='C0', lw=2, label='true')
ax.plot(Xd, Yd, linestyle='none', marker='o', markersize=6, markerfacecolor='w', markeredgecolor='k', label='data')
ax.plot(Xfit, Yfit, lw=2, color='C1', label='cubic spline')
ax.plot(Xfit, Yfit_nat, lw=2, color='C2', label='natural cubic spline')
ax.axvspan(xi[0], xi[1], alpha=0.3, color='grey')
ax.legend()

# %% Figure 5.9
SAVE_FIGURE = True
DPI = 300
DIR_FIGURE = '../figure'
CHAPTER = 'ch5'
PROBLEM = 'splines_lambda'

SEED = 200
np.random.seed(SEED)

# generate samples
def generate_samples(N, f, var=1):
    Xd = np.sort(np.random.random(N))
    Yt = f(Xd)
    Yd = Yt + np.random.normal(scale=var, size=N_sample)
    return Xd, Yd, Yt

N_sample = 100
f = lambda x: np.sin(12*(x+0.2))/(x+0.2)
X = np.linspace(0, 1, 1000)
Y = f(X)
Xd, Yd, _ = generate_samples(N_sample, f)

# splines with different dofs
# my implementation
# basis function of natural spline
df = lambda x, xi, xiK: ((x>xi)*(x-xi)**3 - (x>xiK)*(x-xiK)**3)/(xiK - xi)
natural_spline_base = lambda x, xi: [df(x, xi_, xi[-1]) - df(x, xi[-2], xi[-1]) for xi_ in xi[0:-2]]
natural_spline = lambda x, xi: np.concatenate((np.column_stack([x**i for i in range(2)]), np.column_stack(natural_spline_base(x, xi))), axis=1)

def cal_Omega(xi):
    "Calculate Omega Matrix"
    K = len(xi)
    xiK = xi[-1]
    xiK1 = xi[-2]
    Omega = np.zeros((K, K))
    for j in range(2, K):
        for k in range(2, K):
            xij2 = xi[j-2]
            xik2 = xi[k-2]
            int_jk = trunc_int(xiK, xij2, xik2)
            int_jK1 = trunc_int(xiK, xij2, xiK1)
            int_kK1 = trunc_int(xiK, xik2, xiK1)
            int_K1K1 = trunc_int(xiK, xiK1, xiK1)
            Omega[j,k] = int_jk - int_jK1 - int_kK1 + int_K1K1
    return Omega

def trunc_int(xiK, xia, xib):
    ximax = max(xia, xib)
    f = lambda x: 1/3*x**3 - (xia+xib)/2*x**2 + xia*xib*x
    integral = 36/(xiK-xia)/(xiK-xib)*(f(xiK) - f(ximax))
    return integral

def smooth_spline(X, Y, lambd):
    xi = np.unique(X)
    N = natural_spline(X, xi)
    Omega = cal_Omega(xi)
    S = N@np.linalg.inv(N.T@N + lambd*Omega)@N.T
    Yh = S@Y
    return Yh, S

lambdas = np.array([0.0062, 0.000363, 0.0000355])
Yh = []
dof = []
se = []
xi = np.unique(Xd)
N = natural_spline(Xd, xi)
Omega = cal_Omega(xi)
for lambd in lambdas:
    # S = N@np.linalg.inv(N.T@N + lambd*Omega)@N.T
    yh, S = smooth_spline(Xd, Yd, lambd)
    se.append(np.sqrt((S@S.T).diagonal()))
    Yh.append(yh)
    dof.append(np.trace(S))
    print('Degree of Freedom trace(S) = {}'.format(np.trace(S)))

# csaps implementation
from csaps import csaps
smoothes = 1/(1+lambdas)
Yh_csaps = []
for smooth in smoothes:
    Yh_csaps.append(csaps(Xd, Yd, xi, smooth=smooth))

# plot
nrow = len(lambdas)
fig, axes = plt.subplots(nrow, 1, figsize=(5,5*nrow))
for i, ax in enumerate(axes):
    ax.plot(X, Y, color='C0', lw=2, label='true')
    ax.plot(Xd, Yd, linestyle='none', marker='o', markersize=6, markerfacecolor='w', markeredgecolor='k')
    ax.plot(Xd, Yh[i], color='C1', lw=2, label='my')
    ax.plot(Xd, Yh_csaps[i], linestyle='dashed', color='C2', lw=2, label='csaps')
    ax.fill_between(Xd, Yh[i]-2*se[i], Yh[i]+2*se[i], alpha=0.3, color='C1',zorder=4)
    ax.set_title('df={:4.1f}'.format(dof[i]))
    ax.legend()

# save figure
if SAVE_FIGURE == True:
    utils.save_figure(fig, CHAPTER, PROBLEM)
else:
    plt.show()

#%% CV and EPE
# Notice this is for one sample, and CV could be larger or smaller than EPE
# Better to have several batches of samples, and take the average to estimate the expectation

SAVE_FIGURE = True
DPI = 300
DIR_FIGURE = '../figure'
CHAPTER = 'ch5'
PROBLEM = 'splines_lambda_EPE'

SEED = 150
np.random.seed(SEED)

lambdas = np.logspace(-2, -4.5, 20)

EPE = []
DOF = []
CV = []

Xd, Yd, Yt = generate_samples(N_sample, f)
for lambd in lambdas:
    Yh, S = smooth_spline(Xd, Yd, lambd)
    err  = np.sum((Yh-Yt)**2)/N_sample + var**2
    cv = np.sum(((Yh-Yd)/(1-S.diagonal()))**2)/N_sample
    dof = np.trace(S)
    EPE.append(err)
    CV.append(cv)
    DOF.append(dof)

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(DOF, EPE, lw=2, linestyle='none', marker='o', color='C0', label='EPE')
ax.plot(DOF, CV, lw=2, linestyle='none', marker='o', color='C1', label='CV')
ax.set_xlabel('df$_\lambda$')
ax.set_ylabel('EPE/CV')
ax.legend()

# save figure
if SAVE_FIGURE == True:
    utils.save_figure(fig, CHAPTER, PROBLEM)
else:
    plt.show()