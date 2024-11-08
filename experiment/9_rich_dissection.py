"""More careful dissection of rich MLP"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm, halfnorm
import seaborn as sns
from tqdm import tqdm


import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import SameDifferent 

n_dims = 128
n_points = 10
n_hidden = 1024

noise = 0

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

n_patches = 2

train_task = SameDifferent(noise=0.5, n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(noise=0.5, n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

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
                    # lr=1e-3,
                    # optim=sign_sgd,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# %%
W = np.array(state.params['Dense_0']['kernel'])
a = np.array(state.params['Dense_1']['kernel'])

n_iters = 10_000
xs = np.random.randn(n_iters, 2*n_dims) / np.sqrt(n_dims)
xs += 100 * np.random.randn(*xs.shape)

out = jax.nn.relu(xs @ W) @ a
np.mean(out < 0)

# <codecell>
res = (xs @ W)
idx = 10
plt.hist(res[:,idx], bins=20, density=True)

m = np.mean(res, axis=0)
s = np.std(res, axis=0)

res_sim = norm.rvs(loc=m[idx], scale=s[idx], size=n_iters)
plt.hist(res_sim, density=True, alpha=0.5, bins=20)

# <codecell>
W_p = W[:,a.flatten() > 0]
W_n = W[:,a.flatten() < 0]

res_pre = (xs @ W_n)[:,:].mean(axis=1) / 2
res = jax.nn.relu(xs @ W_n)[:,:].mean(axis=1)

plt.hist(res, bins=20, density=True)
plt.hist(res_pre, bins=20, density=True)
print('mean_pre', np.mean(res_pre))
print('std_pre', np.std(res_pre))
print('mean', np.mean(res))
print('std', np.std(res))

res_pre_sim = norm.rvs(loc=np.mean(res_pre), scale=np.std(res_pre), size=n_iters)
plt.hist(res_pre_sim, alpha=0.5, density=True, bins=20)

dot = xs @ W_n
m = np.mean(dot, axis=0)
s = np.std(dot, axis=0)


res_sim = np.zeros(n_iters)
# for i, (loc, scale) in enumerate(zip(m, s)):
#     samp = norm.rvs(loc=loc, scale=scale, size=n_iters) / W_p.shape[1]
#     # samp = dot[:,i] / W_p.shape[1]
#     res_sim += np.abs(samp) / 2

# k = n_points - 1
k = 4
for i in range(k):
    samp = norm.rvs(loc=np.mean(m), scale=np.mean(s), size=n_iters) / k
    # samp = dot[:,i] / W_p.shape[1]
    res_sim += np.abs(samp) / 2
    # res_sim += samp

# res_sim = res_pre_sim[res_pre_sim > 0]
# res_sim = np.abs(res_pre_sim)
g = plt.hist(res_sim, density=True, bins=20)

print('mean_sim', np.mean(res_sim))
print('std_sim', np.std(res_sim))

# <codecell>
z1s = W_n[:n_dims, :]
overlap = train_task.symbols @ z1s
plt.imshow(overlap[:,:50].T, cmap='plasma', vmin=-1, vmax=1)
plt.colorbar()

# <codecell>
out = np.std(res) / np.std(res_sim)
out

# <codecell>
L = 5
zs = np.random.randn(n_iters, n_dims) * np.sqrt(2/n_dims)

symbols = np.random.randn(L, n_dims) / np.sqrt(n_dims)
symbols.shape

overlap = zs @ symbols.T
out = np.sum(overlap, axis=1)
out = np.abs(out) / 2

plt.hist(out)


# <codecell>
idx = 19
loc, scale = m[idx], s[idx]

samp = norm.rvs(loc=loc, scale=scale, size=n_iters)
true = dot[:,idx]

plt.hist(true, bins=20, density=True)
plt.hist(samp, bins=20, density=True, alpha=0.5)

# <codecell>
res = jax.nn.relu(xs @ W_p).mean(axis=1) - 1.5 * jax.nn.relu(xs @ W_n).mean(axis=1)
np.mean(res < 0)