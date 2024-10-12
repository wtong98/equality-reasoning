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
batch_size = 4096

train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=batch_size)
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

diff = (xs_pos.T @ xs_pos - xs_neg.T @ xs_neg) / np.sqrt(batch_size)
plt.imshow(diff, vmin=-1, vmax=1, cmap='bwr')
plt.colorbar()

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
idx = 2

plt.plot(eig_vec[:,idx])
print(eig_val[idx])

eig_vec[:,idx]

# %%
