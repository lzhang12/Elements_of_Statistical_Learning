"""
Phoneme Recognition using smoothed Logistic Regression

The restricted LR gives a too small beta.

author: zl
update: 2020/07/20
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
PROBLEM = 'phoneme_recognition_smooth_spline'

SEED = 200
np.random.seed(SEED)

#%%
# read in data
data = pd.read_csv('../data/phoneme.csv', index_col=0)

# choose phonemes
phonemes = ['aa', 'ao']

X = []
for phoneme in phonemes:
    x = data[data['g']==phoneme][[i for i in data.columns if i.startswith('x')]].values
    X.append(x)
    print('Number of {} samples = {}'.format(phoneme, x.shape[0]))

# log-periodogram
N_sample = 15
Xp = []
for x in X:
    id_ = np.random.choice(np.arange(x.shape[0]), N_sample, replace=False)
    Xp.append(x[id_,:])

freqs = np.arange(1, 257)
colors = ['C0', 'C1']
hdls = []
fig, ax = plt.subplots(1,1,figsize=(8,4))

for i, xp in enumerate(Xp):
    hdl = ax.plot(freqs, xp.T, colors[i], lw=0.5)
    hdls.append(hdl[0])
ax.set_xlabel('Frequency')
ax.set_ylabel('Log-periodogram')
ax.legend(hdls, phonemes)

# %% Classification using raw LR
# phoneme 0 ==> y=0
# phoneme 1 ==> y=1
X_arr = np.concatenate(X)
y_arr = np.concatenate([np.ones(a.shape[0])*i for i, a in enumerate(X)])

Xy_arr = np.column_stack((X_arr, y_arr))

N_tr = 1000
id_tr = np.random.choice(Xy_arr.shape[0], N_tr, replace=False)
id_te = np.delete(np.arange(Xy_arr.shape[0]), id_tr)
Xy_tr = Xy_arr[id_tr,:]
Xy_te = Xy_arr[id_te,:]
X_tr, y_tr = Xy_tr[:,:-1], Xy_tr[:,-1]
X_te, y_te = Xy_te[:,:-1], Xy_te[:,-1]

LR = LogisticRegression(penalty='none', solver='newton-cg')
beta = np.squeeze(LR.fit(X_tr, y_tr).coef_)

# %% Classification using restricted LR
# Smoothed coefficients using natural cubic splines
N_sp = 12
xi = np.linspace(1, 256, N_sp)

# natural cubic spline basis
df = lambda x, xi, xiK: ((x>xi)*(x-xi)**3 - (x>xiK)*(x-xiK)**3)/(xiK - xi)
natural_spline_base = lambda x, xi: [df(x, xi_, xi[-1]) - df(x, xi[-2], xi[-1]) for xi_ in xi[0:-2]]
natural_spline = lambda x, xi: np.concatenate((np.column_stack([x**i for i in range(2)]), np.column_stack(natural_spline_base(x, xi))), axis=1)

H = natural_spline(freqs, xi)
X_tr_sm = X_tr@H
LR_sm = LogisticRegression(penalty='none', solver='newton-cg', max_iter=10000, verbose=10)
LR_sm_bfgs = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000, verbose=10)
theta = np.squeeze(LR_sm.fit(X_tr_sm, y_tr).coef_)
LR_sm_bfgs.fit(X_tr_sm, y_tr)
beta_sm = H@theta

#%% Compare Training and Test Error
tr = LR.score(X_tr, y_tr)
tr_sm = LR_sm.score(X_tr_sm, y_tr)
tr_sm_bfgs = LR_sm_bfgs.score(X_tr_sm, y_tr)

te = LR.score(X_te, y_te)
X_te_sm = X_te@H
te_sm = LR_sm.score(X_te_sm, y_te)
te_sm_bfgs = LR_sm_bfgs.score(X_te_sm, y_te)

tab = [('Training Error', 1-tr, 1-tr_sm, 1-tr_sm_bfgs), ('Test Error', 1-te, 1-te_sm, 1-te_sm_bfgs)]
print(tabulate(tab, headers=['Raw', 'Regularized \n (newton-cg)', 'Regularized \n (bfgs)'], tablefmt='fancy_grid'))

# %% plot beta & theta
freqs = np.arange(1, 257)
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(freqs, beta, 'C0', lw=0.5)
ax.plot(freqs, beta_sm, 'C1', lw=2)
ax.set_xlabel('Frequency')
ax.set_ylabel('LR Coefficients')
ax.set_ylim([-0.4, 0.4])

# %%
if SAVE_FIGURE == True:
    utils.save_figure(fig, CHAPTER, PROBLEM)
else:
    plt.show()

# %%
