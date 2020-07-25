"""
Smooth Spline Fit of the Spinal BMD data for male and female
"""

#%%
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

plt.style.use('../utils/default_plot_style.mplstyle')
np.set_printoptions(precision=3)

SAVE_FIGURE = True
CHAPTER = 'ch5'
PROBLEM = 'spinal_BMD'

SEED = 200
np.random.seed(SEED)

# %% data
data = pd.read_csv('../data/BMD.csv', index_col=0, sep='\s+')

Xm = data[data['gender']=='male'][['age','spnbmd']].sort_values(by=['age']).values
Xf = data[data['gender']=='female'][['age','spnbmd']].sort_values(by=['age']).values

Xm, ym = Xm[:,0], Xm[:,1]
Xf, yf = Xf[:,0], Xf[:,1]

# %% smooth spline fit
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
            # int_jk = trunc_int(xi, j-2, k-2)
            # int_jK1 = trunc_int(xi, j-2, K-2)
            # int_kK1 = trunc_int(xi, k-2, K-2)
            # int_K1K1 = trunc_int(xi, K-2, K-2)
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

# smoother matrix
lambda_ = 0.00022
yh = []
for X,y in zip([Xm, Xf], [ym, yf]):
    xi = np.unique(X)
    N = natural_spline(X, xi)
    Omega = cal_Omega(xi)
    # https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/blob/master/chapters/5%20Basis%20Expansions%20and%20Regularization/4%20Smoothing%20Splines.ipynb    S = N@np.linalg.inv(N.T@N + lambda_*3600*Omega)@N.T
    # where is this 3600 coming from ?
    S = N@np.linalg.inv(N.T@N + lambda_*3600*Omega)@N.T
    yh_ = S@y
    yh.append(yh_)

print('Degree of Freedom trace(S) = {}'.format(np.trace(S)))

# %% plot
fig, ax = plt.subplots(1,1,figsize=(7,5))

ax.plot(Xm, ym, color='C0', linestyle='none', marker='o', markersize=4, markerfacecolor='C0', label='male')
ax.plot(Xf, yf, color='C1', linestyle='none', marker='o', markersize=4, markerfacecolor='C1', label='female')
ax.plot(Xm, yh[0], color='C0')
ax.plot(Xf, yh[1], color='C1')
ax.hlines(0, *ax.get_xlim(), linestyle='dashed', color='grey', lw=0.5)
ax.set_xlabel('age')
ax.set_ylabel('Relative Change in Spinal BMD')
ax.legend()

# %% save figure
if SAVE_FIGURE == True:
    utils.save_figure(fig, CHAPTER, PROBLEM)
else:
    plt.show()
