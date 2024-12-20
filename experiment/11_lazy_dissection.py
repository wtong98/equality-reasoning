"""More careful dissection of lazy MLP"""

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
n_dims = 128
n_points = 256
n_hidden = 4096

gamma0 = 1e-6
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0**2 * 10

n_patches = 2

train_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

config = MlpConfig(mup_scale=False,
                   as_rf_model=True,
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
                    train_iters=1,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)


state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=50_000,
                    # optim=optax.sgd,
                    # lr=lr,
                    gamma=gamma,
                    seed=None)

# <codecell>
xs, ys = next(test_task)
out = state.apply_fn({'params': state.params}, xs)
preds = (out > 0).astype(bool)

print(np.mean(ys == preds))
print('---')
print(np.mean(ys[ys>0] == preds[ys>0]))
print(np.mean(ys[ys==0] == preds[ys==0]))

print('---')
print(np.mean(ys[preds>0] == preds[preds>0]))
print(np.mean(ys[preds==0] == preds[preds==0]))

print('---')
print(np.mean(ys))
print(np.mean(preds))


# <codecell>
### VALIDATE WITH EXPERIMENTS
df = collate_dfs('remote/11_lazy_dissection/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]
    hist_loss = [m['loss'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].noise,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        min(hist_loss),
    ], index=['name', 'gamma0', 'sig2', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen', 'acc_best', 'loss_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
def pred_pos_acc(L, sig2, neg_dim_guess_prefac=0):
    a = 1.33**2  # NOTE: a seems to be an effectively tunable free parameter (not good)

    neg_dim_guess = neg_dim_guess_prefac * (1 + 2*sig2)
    prefactor = np.sqrt(2 / (np.pi - 2))

    t1 = np.sqrt(sig2 + 1) - a * np.sqrt(sig2 + neg_dim_guess)
    t2 = np.sqrt((a**2 + 1) * sig2 + 1 + a**2 * neg_dim_guess)
    t = t1 / t2

    z = prefactor * t * np.sqrt(L)
    return norm.cdf(z)

def pred_neg_acc(L, sig2=0, l_adjust=1):  # TODO: incorporate sig2
    a_raw = 1.5**2
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = np.sqrt(L - l_adjust) * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)

def pred_acc(L, sig2):
    if L == 2 and sig2 <= 0.1:
        return 0.75  # TODO: refine

    pos_acc = pred_pos_acc(L, sig2)
    neg_acc = pred_neg_acc(L, sig2)

    return (pos_acc + neg_acc) / 2


Ls = np.unique(plot_df['n_symbols'])
sig2s = np.unique(plot_df['sig2'])

res = []
for L, sig2 in itertools.product(Ls, sig2s):
    res.append({
        'name': 'prediction',
        'n_symbols': L,
        'sig2': sig2,
        'acc_best': pred_acc(L, sig2),
        'acc_unseen': pred_acc(L, sig2)
    })

res_df = pd.DataFrame(res)
res_df

# <codecell>
# mdf = pd.concat((plot_df, res_df))
mdf = plot_df.copy()
mdf = mdf[(mdf['sig2'] == 0) & (mdf['name'] == 'RF')]


gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', col='n_dims', col_wrap=4, hue='name', marker='o', kind='line')
gs.set(xscale='log')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['sig2'] == 0) 
    & (mdf['name'] == 'RF')
    & (mdf['n_dims'] == 32)
    ]

gs = sns.lineplot(mdf, x='n_symbols', y='acc_unseen', hue='name', marker='o')
gs.set(xscale='log')

xs = np.unique(np.sort(mdf['n_symbols']))
preds = 1 - np.exp(-(xs / 32)**2)

plt.plot(xs, preds)

