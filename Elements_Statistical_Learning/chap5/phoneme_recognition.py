"""
Phoneme Recognition using smoothed Logistic Regression

The restricted LR gives a too small beta.

author: zl
update: 2020/07/20
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('../utils/default_plot_style.mplstyle')
np.set_printoptions(precision=3)

SAVE_FIGURE = True
DPI = 300
DIR_FIGURE = '../figure'
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
from sklearn.linear_model import LogisticRegression
# phoneme 0 ==> y=0
# phoneme 1 ==> y=1
X_arr = np.concatenate(X)
y_arr = np.concatenate([np.ones(a.shape[0])*i for i, a in enumerate(X)])

Xy_arr = np.column_stack((X_arr, y_arr))

N_tr = 1000
id_ = np.random.choice(Xy_arr.shape[0], N_tr, replace=False)
Xy_tr = Xy_arr[id_,:]
X_tr = Xy_tr[:,:-1]
y_tr = Xy_tr[:,-1]

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
Xst = X_tr@H
LR = LogisticRegression(penalty='none', solver='newton-cg')
theta = np.squeeze(LR.fit(Xst, y_tr).coef_)
beta_st = H@theta

# %% plot beta & theta
freqs = np.arange(1, 257)
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(freqs, beta, 'C0', lw=0.5)
ax.plot(freqs, beta_st, 'C1', lw=2)
ax.set_xlabel('Frequency')
ax.set_ylabel('LR Coefficients')
ax.set_ylim([-0.4, 0.4])

# %%
if SAVE_FIGURE == True:
    fn = os.path.join(DIR_FIGURE, '_'.join([CHAPTER, PROBLEM]))
    fig.savefig(fn, dpi=DPI)
else:
    plt.show()

# %%
