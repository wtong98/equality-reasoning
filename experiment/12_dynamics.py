"""Reverse-engeering the learning dynamics"""

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

# <codecell>
n_dims = 64
n_points = 32
n_hidden = 1024

test_every = 100

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0**2 * 10

n_patches = 2

train_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

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
                    test_every=test_every,
                    test_iters=1,
                    train_iters=30_000,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None,
                    save_params=True)


# %%
aa = []
Ws = []

for params in hist['params']:
    Ws.append(params['Dense_0']['kernel'])
    aa.append(params['Dense_1']['kernel'])

aa = np.array(aa).squeeze()
Ws = np.array(Ws)

# %%
sidxs = np.argsort(state.params['Dense_1']['kernel'].squeeze())
times = np.arange(0, len(aa)) * test_every
times[0] = 1

plt.plot(times, aa[:,2], '--o')
plt.plot(times, aa[:,sidxs[-1]], '--o')
# plt.xscale('log')
# plt.yscale('log')

plt.plot(times, 0.49 * np.log(times))

from scipy.special import lambertw
plt.plot(times, 0.62 * lambertw(times))

# plt.xscale('log')
# plt.yscale('log')

c = 0.01 * times
pred = np.exp(c * (times - lambertw(np.exp(2 * c * times))))
plt.plot(times, pred)

# <codecell>
plt.plot(times, np.linalg.norm(Ws[:,:,sidxs[0]], axis=1))
# plt.plot(times, np.log(times))
# plt.plot(times, lambertw(times))

plt.plot(times[1:], 10 - 100 / times[1:]**0.5)

# plt.xscale('log')
# plt.yscale('log')



# <codecell>
xs, _ = next(train_task)
xs = xs.reshape(xs.shape[0], -1)

fs = [np.mean(np.abs(jax.nn.relu(xs @ Ws[i]) @ aa[i][:,None])) for i in range(len(aa))]
fs = np.array(fs)

# <codecell>
# plt.plot(times, aa[:,1])
# plt.plot(times, 1 - np.exp(-lambertw(np.exp(- 0.001 * times))))

plt.plot(times, 0.04 * fs)
plt.plot(times, np.linalg.norm(Ws[:,:,sidxs[0]], axis=1))
plt.plot(times, np.abs(aa[:,sidxs[0]]))
# plt.plot(times, 1 / (1 + np.exp(np.abs(0.01 * fs))))
# plt.plot(times, 1 / (1 + np.exp(0.01) * times))

# plt.xscale('log')
# plt.yscale('log')

# plt.plot(times, 1.1 * np.log(times))
plt.plot(times, np.log(2 * times + np.exp(0.14)))

# <codecell>

from scipy.integrate import solve_ivp

def f(t, a):
    return 0.35 * a * np.exp(-a)

res = solve_ivp(f, (0, 30000), np.array([0.14]))

plt.plot(res.t, res.y.flatten(), '--o')

# <codecell>
plt.plot(times, np.exp(-fs))
plt.plot(times, 1 / (1 + np.exp(fs)))

# plt.yscale('log')


# %%
plt.hist(aa[0,:], bins=50)
plt.hist(aa[-1,:], bins=50, alpha=0.5)

# pred = aa[0,:] * 0.01 * np.log(5000)
# plt.hist(pred)
