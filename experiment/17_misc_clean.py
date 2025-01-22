"""Miscellaneous illustrative figures"""

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

set_theme()

# <codecell>
n_points = 512
n_dims = 128
n_hidden = 128

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

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
                    train_iters=15_000, 
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
# plt.rcParams.update({'font.size': 14})
plt.gcf().set_size_inches(3.5, 2.5)

plt.scatter(a[sort_idxs], cos_dists)
plt.xlabel('$a_i$')
plt.ylabel(r'$(\mathbf{v}_i^1 \cdot \mathbf{v}_i^2)\, / \, \ell_i$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig/ccn/rich_weight_struct.svg')

# note: test acc attained by lazy model is ~0.7

# <codecell>
### PSVRT EXAMPLES
task = SameDifferentPsvrt(patch_size=5, n_patches=3, batch_size=10, seed=3)
xs, ys = next(task)

plt.imshow(xs[3], cmap='binary')
plt.axis('off')
plt.savefig('fig/ccn/psvrt_same.svg')

# <codecell>
plt.imshow(xs[1], cmap='binary')
plt.axis('off')
plt.savefig('fig/ccn/psvrt_diff.svg')

# <codecell>
### PENTOMINO EXAMPLES
task = SameDifferentPentomino(width=2, batch_size=10)
xs, ys = next(task)

xs = np.clip(xs + 0.4, a_max=1, a_min=0)
plt.imshow(xs[0], cmap='binary', vmin=0)
plt.axis('off')
plt.savefig('fig/ccn/pentomino_same.svg')

# <codecell>
plt.imshow(xs[1], cmap='binary', vmin=0)
plt.axis('off')
plt.savefig('fig/ccn/pentomino_diff.svg')

# <codecell>
### CIFAR-100 examples
task = SameDifferentCifar100(np.arange(100), batch_size=50, preprocess_cnn=False, normalize=False)
xs, ys = next(task)

xs = xs.reshape(xs.shape[0], 2, 32, 32, 3, order='F')
xs = np.swapaxes(xs, 2, 3)
xs.shape

xs_sel = xs[0]
print(ys[0])

xs_sel = np.concat(xs_sel, axis=1)

plt.imshow(xs_sel)
plt.axis('off')
plt.savefig('fig/ccn/cifar100_same.svg')

# %%
idx = 19
xs_sel = xs[idx]
print(ys[idx])

xs_sel = np.concat(xs_sel, axis=1)

plt.imshow(xs_sel)
plt.axis('off')
plt.savefig('fig/ccn/cifar100_diff.svg')

# <codecell>
n_points = 256
n_hidden = 512

gamma0 = 1e-5
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 1

inc_set = gen_patches(patch_size=5, n_examples=n_points)
train_task = SameDifferentPsvrt(patch_size=5, n_patches=2, inc_set=inc_set)
test_task = SameDifferentPsvrt(patch_size=5, n_patches=2)

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
                    train_iters=15_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# note: attained lazy attained test accuracy around 0.85
# <codecell>
W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
idx = 0

plt.imshow(W_sort[:,idx].reshape(10, 10), vmin=-0.1, vmax=0.1, cmap='viridis')
plt.axhline(y=4.5, color='white')
plt.axvline(x=4.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$')
plt.axis('off')

plt.savefig('fig/ccn/concept/psvrt_lazy_neg.svg')

# <codecell>
idx = -3

plt.imshow(W_sort[:,idx].reshape(10, 10), vmin=-0.1, vmax=0.1, cmap='viridis')
plt.axhline(y=4.5, color='white')
plt.axvline(x=4.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$')
plt.axis('off')

plt.savefig('fig/ccn/concept/psvrt_lazy_pos.svg')

# <codecell>
### PENTOMINOS
n_points = 16
n_hidden = 512

gamma0 = 1e-5
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

train_task = SameDifferentPentomino(ps=np.arange(n_points), blur=0.5, random_blur=True)
test_task = SameDifferentPentomino(ps=[16, 17], blur=0, random_blur=False)

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
                    train_iters=15_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# attains test acc near chance
# %%
W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
idx = 0

plt.imshow(W_sort[:,idx].reshape(14, 14), vmin=-0.1, vmax=0.1, cmap='viridis')
plt.axhline(y=6.5, color='white')
plt.axvline(x=6.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$')
plt.axis('off')

plt.savefig('fig/ccn/concept/pentomino_lazy_neg.svg')

# <codecell>
idx = -1

plt.imshow(W_sort[:,idx].reshape(14, 14), vmin=-0.1, vmax=0.1, cmap='viridis')
plt.axhline(y=6.5, color='white')
plt.axvline(x=6.5, color='white')
plt.colorbar()

plt.title(f'$a = {a[sort_idxs[idx]]:.2f}$')
plt.axis('off')

plt.savefig('fig/ccn/concept/pentomino_lazy_pos.svg')

# <codecell>
