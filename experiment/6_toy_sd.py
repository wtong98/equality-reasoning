"""Simple SD task"""


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

# <codecell>
df = collate_dfs('remote/6_toy_sd/data_div', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]
    hist_loss = [m['loss'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen', 'best_acc', 'best_loss'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
g = sns.lineplot(plot_df, x='n_symbols', y='best_acc', hue='name', marker='o')
g.set_xscale('log', base=2)

# <codecell>
mdf = plot_df[(plot_df['gamma0'] <= -2) & (plot_df['name'] != 'MLP (Adam)')]
g = sns.lineplot(mdf, x='n_symbols', y='best_loss', hue='name', marker='o')
g.set_xscale('log', base=2)
g.set_yscale('log')

# <codecell>
plot_df['name'].str.contains("gamma")


# <codecell>
mdf = plot_df[plot_df['n_dims'] == 128]
g = sns.lineplot(mdf, x='n_symbols', y='acc_unseen', hue='name', marker='o')

g.set_xscale('log', base=2)

# <codecell>
threshold = 0.75
all_mdfs = []
for mdf in plot_dfs:
    mdf = mdf[mdf['acc_unseen'] > threshold]
    mdf = mdf.groupby(['name', 'n_dims']).min()
    all_mdfs.append(mdf)

mdf = pd.concat(all_mdfs[:2])
mdf

# <codecell>
g = sns.lineplot(mdf, x='n_dims', y='n_symbols', hue='name', marker='o')
g.set_xscale('log', base=2)
g.set_yscale('log', base=2)
# <codecell>
# NOTE: dimension dependence seems to enter when considering patch sizes > 2
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
                   as_rf_model=True,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

# config = SimpleTransformerConfig(n_hidden=n_hidden, gamma=gamma)
# config = SimpleTransformerConfig(n_hidden=n_hidden)

# config = TransformerConfig(n_layers=1,
#                            n_hidden=512,
#                            pos_emb=False,
#                            n_mlp_layers=0,
#                            n_heads=2,
#                            layer_norm=False,
#                            as_rf_model=False,
#                            residual_connections=False,
#                            use_simple_att=True,
#                            freeze_emb=False)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=25_000, 
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# <codecell>
jax.tree.map(jnp.shape, state.params)

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel']


idx = 41

w_sel = W[:,idx].reshape(8, -1)
plt.imshow(w_sel @ w_sel.T, vmin=-5, vmax=5, cmap='bwr')
plt.colorbar()

a[idx]

# <codecell>
w_proj = w_sel @ train_task.symbols.T
plt.imshow(w_proj[:,:50], cmap='bwr')
# plt.colorbar()

# <codecell>
plt.hist(a.flatten(), bins=50)


# <codecell>
xs, ys = next(test_task)
out = state.apply_fn({'params': state.params}, xs)
preds = (out > 0).astype(bool)

print(np.mean(ys[ys>0] == preds[ys>0]))
print(np.mean(ys[ys==0] == preds[ys==0]))

print('---')
print(np.mean(ys[preds>0] == preds[preds>0]))
print(np.mean(ys[preds==0] == preds[preds==0]))

print('--')
print(np.mean(ys))
print(np.mean(preds))


# <codecell>
### COORDINATE CHECKING
all_norms = []

# for n_hidden in [16, 64, 256, 1024]:
for n_patches, n_hidden in zip([2, 4, 8, 16], [16, 64, 256, 1024]):
    n_points = 8
    n_dims = 128
    # n_hidden = 512

    gamma0 = 0.01
    gamma = gamma0 * np.sqrt(n_hidden)
    # lr = gamma**2 * 0.1
    lr = gamma0**2 * 0.01

    train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
    test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

    config = MlpConfig(mup_scale=True,
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
                        gamma=gamma,
                        test_every=1000,
                        train_iters=1, 
                        optim=optax.sgd,
                        lr=lr,
                        seed=None)

    xs, _ = next(train_task)
    xs = xs.reshape(xs.shape[0], -1)

    jax.tree.map(jnp.shape, state.params)

    W = state.params['Dense_0']['kernel']
    a = state.params['Dense_1']['kernel']
    
    act = xs @ W
    norm = np.mean(np.linalg.norm(act, axis=1))
    all_norms.append(norm)

# <codecell>
plt.plot(all_norms, '--o')
# %%
### VALIDATION OF RF CALCULATIONS <-- TODO: need to complete (should be correct, 
# but need to address inconsistencies). May be issue of normalization -- pay attention to how its done exactly
n_points = 2048
n_hidden = 512
# n_dimss = np.round(np.linspace(16, 512, num=10)).astype(int)
# n_dimss = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
n_dimss = np.array([2, 4, 8, 16, 32, 64, 128])

all_avgs = []
all_avgs_n = []

for n_dims in n_dimss:
    xs_pos = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
    xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
    # xs_pos = np.sqrt(np.sqrt(2)) * np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
    xs_neg = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)

    W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)
    hp = jax.nn.relu(xs_pos @ W)

    kp = hp @ hp.T
    kp = np.triu(kp, k=1)
    # kp = kp[np.nonzero(kp)]

    hn = jax.nn.relu(xs_neg @ W)

    kn = hn @ hn.T
    kn = np.triu(kn, k=1)
    # kn = kn[np.nonzero(kn)]

    avg = np.mean(kp)

    all_avgs.append(avg)
    all_avgs_n.append(np.mean(kn))

plt.loglog(n_dimss, all_avgs, '--o')
plt.loglog(n_dimss, all_avgs_n, '--o')
# plt.loglog(n_dimss, 80 / n_dimss)
# plt.loglog(n_dimss, 80 * np.sqrt(2) / n_dimss)

# <codecell>
n_points = 1_000
n_dims = 64
n_hidden = 10_000

# task = SameDifferent(n_symbols=None, n_dims=n_dims, batch_size=n_points)
# xs, ys = next(task)
# xs = xs.reshape(xs.shape[0], -1)
# xs_pos = xs[ys==1]
# xs_neg = xs[ys==0]

xs_pos = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
# xs_pos = np.sqrt(np.sqrt(2)) * np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
xs_neg = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)

# W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)
W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)

xs_pos = jax.nn.relu(xs_pos @ W)
xs_neg = jax.nn.relu(xs_neg @ W)

dp = np.triu(xs_pos @ xs_pos.T, k=1) / n_hidden
dp = dp[np.nonzero(dp)]
dn = np.triu(xs_neg @ xs_neg.T, k=1) / n_hidden
dn = dn[np.nonzero(dn)]

print(np.var(dp))
print(2*np.var(dn))

print(np.mean(dp))
print(np.mean(dn))

# print(np.mean(dp**2))
# print(np.mean(dn**2))

# dp = dp + dp**2/np.pi
# dn = dn + dn**2/np.pi

plt.hist(dp, bins=50, density=True, alpha=0.7)
plt.hist(dn, bins=50, density=True, alpha=0.7)

plt.axvline(x=np.mean(dp))
plt.axvline(x=np.mean(dn), color='red')
# plt.hist(dn * np.sqrt(2), bins=50, density=True, alpha=0.3)


# <codecell>
all_dp = []
all_dn = []

W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)

for _ in range(n_points):
    xs_pos = np.random.randn(2, 2*n_dims) / np.sqrt(n_dims)
    xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
    xs_neg = np.random.randn(2, 2*n_dims) / np.sqrt(n_dims)

    xs_pos = jax.nn.relu(xs_pos @ W) / n_hidden
    xs_neg = jax.nn.relu(xs_neg @ W) / n_hidden

    dp = np.triu(xs_pos @ xs_pos.T, k=1)
    dp = dp[np.nonzero(dp)]
    dn = np.triu(xs_neg @ xs_neg.T, k=1)
    dn = dn[np.nonzero(dn)]

    all_dp.append(dp[0])
    all_dn.append(dn[0])


# <codecell>
plt.hist(all_dp, bins=50, alpha=0.7)
plt.hist(all_dn, bins=50, alpha=0.7)

plt.axvline(x=np.mean(all_dp))
plt.axvline(x=np.mean(all_dn), color='red')

# <codecell>
# u = xs_pos @ xs_pos.T / np.linalg.norm(xs_pos, axis=-1)**2
# u = u * (1 - 1/np.pi * np.arccos(u)) * (1 / np.pi) * np.sqrt(1 - u**2)

# plt.imshow(u)
