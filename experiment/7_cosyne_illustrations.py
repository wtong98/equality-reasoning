"""Tidy illustrations for Cosyne figures"""

# <codecell>
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import * 

# <codecell>
### PCA and Same Different recovers parallel / anti-parallel features
n_dims = 1024
n_points = 32
batch_size = 3000

train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=batch_size)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

train_task.symbols.shape

xs, ys = next(train_task)
xs = xs @ train_task.symbols.T
xs = xs.reshape(xs.shape[0], -1)

pos_idx = ys.astype(bool)

xs_pos = xs[pos_idx]
xs_neg = xs[~pos_idx]

plt.gcf().set_size_inches(2, 2)
diff = (xs_pos.T @ xs_pos - xs_neg.T @ xs_neg) / np.sqrt(batch_size)
plt.imshow(diff, vmin=-1, vmax=1, cmap='bwr')
plt.colorbar()
plt.gca().set_axis_off()
plt.title(r'$X_1^T X_1 - X_0^T X_0$')

plt.savefig('fig/cosyne/xx_true.svg')

# <codecell>
plt.gcf().set_size_inches(2, 2)
diff = np.zeros((64, 64))
diff[0:32,32:64] = np.eye(32)
diff[32:64,0:32] = np.eye(32)
plt.imshow(diff, vmin=-1, vmax=1, cmap='bwr')

plt.colorbar()
plt.gca().set_axis_off()
plt.savefig('fig/cosyne/xx_ideal.svg')

# <codecell>
### PARALLEL / ANTI-PARALLEL in SD

n_points = 512
n_dims = 16
# n_hidden = 892
n_hidden = 128

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 1

n_patches = 2

train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

config = MlpConfig(mup_scale=True,
                   as_rf_model=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=5_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# %%
jax.tree.map(np.shape, state.params)

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
w1, w2 = W_sort[:n_dims], W_sort[n_dims:]
dots = w1.T @ w2 / (np.linalg.norm(w1, axis=0) * np.linalg.norm(w2, axis=0))
cos_dists = np.diag(dots)

# <codecell>
plt.rcParams.update({'font.size': 14})
plt.gcf().set_size_inches(3, 2)
plt.scatter(a[sort_idxs], cos_dists)
plt.xlabel('$a$')
plt.ylabel(r'$\cos(\theta)$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig/cosyne/angles.svg')


# <codecell>
### PENTOMINO SD VIZ
n_hidden = 512

width = 2
ps = np.random.permutation(np.arange(18))
n_train = 18

ps_train = ps
ps_test = ps

train_task = SameDifferentPentomino(ps=ps_train, width=width, batch_size=128, blur=0)
test_task = SameDifferentPentomino(ps=ps_test, width=width, batch_size=128, blur=0)


gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 1

config = MlpConfig(n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu',
                   )

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=50_000, 
                    optim=optax.sgd,  # NOTE: sharp contrast in using adam vs sgd
                    seed=None,
                    gamma=gamma,
                    lr=lr)


# %%
W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
idxs = [0, 1, -2, -1]

fig, axs = plt.subplots(2, 2)
for idx, ax in zip(idxs, axs.ravel()):
    ax.imshow(W_sort[:,idx].reshape(14, 14))
    ax.set_axis_off()
    ax.set_title(rf'$a = {a[sort_idxs[idx]]:.2f}$')

fig.tight_layout()
fig.savefig('fig/cosyne/pentomino_ws.svg')
