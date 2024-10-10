"""
Reverse engineering success / failure conditions
"""

# <codecell>
from pathlib import Path

from flax import traverse_util
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
from model.transformer import TransformerConfig, SimpleTransformerConfig
from task.same_different import SameDifferent 
from task.ti import TiTask


n_points = 16
n_dims = 128
n_hidden = 512

gamma = 1_000_000
# lr = gamma**2 * 0.1
lr = gamma * 0.1

train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)


train_task.symbols.shape

xs, ys = next(test_task)

# <codecell>
z_uns = xs @ train_task.symbols.T

z_uns = z_uns.reshape(1024, -1)
z_uns_inv = np.linalg.pinv(z_uns.T)

z_reco = z_uns @ np.linalg.pinv(z_uns)
plt.imshow(z_reco[:256,:256])
plt.colorbar()

# plt.imshow(np.linalg.pinv(z_uns) @ z_uns)

# <codecell>
train_task = TiTask(n_symbols=10, sep_dists=[1])
test_task = TiTask(n_symbols=10, sep_dists=[2, 3])

z_sn, _ = next(train_task)
z_uns, _ = next(test_task)

# <codecell>
z_uns = z_uns.reshape(z_uns.shape[0], -1)
z_reco = z_uns @ np.linalg.pinv(z_uns)
z_reco.shape

plt.imshow(z_reco)
plt.colorbar()

# <codecell>
evals, evecs = np.linalg.eig(z_reco)
plt.plot(evals)

# <codecell>
z_uns = xs @ train_task.symbols.T
z_uns = z_uns.reshape(-1, 16)
z_uns.shape

z_reco = z_uns @ np.linalg.pinv(z_uns)
plt.imshow(z_reco[:256, :256])
plt.colorbar()

# <codecell>
train_task = TiTask(n_symbols=10, sep_dists=[1])
test_task = TiTask(n_symbols=10, sep_dists=[2, 3])

z_sn, _ = next(train_task)
z_uns, _ = next(test_task)

# <codecell>
z_uns = z_uns.reshape(-1, z_uns.shape[-1])
z_reco = z_uns @ np.linalg.pinv(z_uns)
z_reco.shape

plt.imshow(z_reco)
plt.colorbar()

# %%
evals, evecs = np.linalg.eig(z_reco)

evals

# <codecell>
### PCA and Same Different recovers parallel / anti-parallel features
n_dims = 1024
n_points = 16

train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=512)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

train_task.symbols.shape

xs, ys = next(train_task)
xs = xs @ train_task.symbols.T
xs = xs.reshape(xs.shape[0], -1)


pos_idx = ys.astype(bool)

xs_pos = xs[pos_idx]
xs_neg = xs[~pos_idx]

# plt.imshow(xs_pos.T @ xs_pos)
# plt.imshow(xs_neg.T @ xs_neg)

diff = xs_pos.T @ xs_pos - xs_neg.T @ xs_neg
# plt.imshow(diff)

eig_val, eig_vec = np.linalg.eig(diff)

# <codecell>
# plt.imshow(diff)
idx = 8

plt.plot(eig_vec[:,idx])
print(eig_val[idx])

# <codecell>
diff = np.zeros((16, 16))
diff[0:8,8:16] = np.eye(8)
diff[8:16,0:8] = np.eye(8)
plt.imshow(diff)

eig_val, eig_vec = np.linalg.eig(diff)

# <codecell>
# plt.plot(np.abs(eig_val))
idx = 15

plt.plot(eig_vec[:,idx])
print(eig_val[idx])
