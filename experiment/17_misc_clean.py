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
# n_hidden = 892
n_hidden = 128

gamma0 = 1e-5
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

n_patches = 2

train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

config = MlpConfig(mup_scale=False,
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
plt.savefig('fig/ccn/lazy_weight_struct.svg')

# note: test acc attained by lazy model is ~0.65