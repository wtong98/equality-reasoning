"""Comparing the impact of architecture on same-different performance"""


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
from model.transformer import TransformerConfig
from task.same_different import SameDifferent 
from task.ti import TiTask

# <codecell>
df = collate_dfs('remote/1_compare_arch/generalize')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
grid = sns.relplot(plot_df, x='n_symbols', y='acc_seen', hue='name', col='n_dims', col_wrap=3, height=2)
grid.set(xscale='log')
plt.savefig('fig/compare_arch_seen.png')

# <codecell>
df = collate_dfs('remote/1_compare_arch/generalize_rf')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
grid = sns.relplot(plot_df, x='n_symbols', y='acc_seen', hue='name', col='n_dims', col_wrap=3, height=2)
grid.set(xscale='log')
plt.savefig('fig/compare_arch_rf_seen.png')

# <codecell>
n_points = 256
n_dims = 32
n_hidden = 512

# NOTE: relu is enough to omit bias from initial layer! (What's happening exactly? How many heads do we need?) <-- STOPPED HERE

# sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
# test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

train_task = TiTask(n_symbols=6, sep_dists=[1])
test_task = TiTask(n_symbols=6, sep_dists=[2, 3, 4])

config = MlpConfig(mup_scale=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                #    feature_learning_strength=gamma0,
                   use_bias=False,
                   act_fn='relu')

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
                    loss='mse',
                    gamma=None,
                    test_every=1000,
                    train_iters=5_000, 
                    seed=None)

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

# %%
xs_same = np.random.randn(2, n_dims) / np.sqrt(n_dims)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, n_dims) / np.sqrt(n_dims)

xs = np.stack([xs_same, xs_diff])

out, feats = state.apply_fn({'params': state.params}, xs, mutable='intermediates')
# out, feats = state.apply_fn({'params': state.params}, xs, capture_intermediates=True)
# feats
print(out)

np.round(feats['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['attention_weights'], decimals=3)

# %%
xs_same = np.random.randn(2, 512) / np.sqrt(512)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, 512) / np.sqrt(512)

# M = np.random.randn(32, 32) / np.sqrt(32)
b = np.random.randn(512) / np.sqrt(512) * 0
M = m[0]
# b = np.ones(32) * 1000
# b = np.ones(32) * 100

# xs_same @ M @ xs_same.T

xs_diff @ M @ xs_diff.T + (xs_diff[0] @ M @ b + xs_diff[1] @ M.T @ b)

# %%
xs_same = np.random.randn(2, n_dims) / np.sqrt(n_dims)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, n_dims) / np.sqrt(n_dims)

W = state.params['Dense_0']['kernel']
xs_same = xs_same @ W + state.params['Dense_0']['bias']
xs_diff = xs_diff @ W + state.params['Dense_0']['bias']

# xs_same = xs_same @ W
# xs_diff = xs_diff @ W

jax.tree.map(np.shape, state.params)
k = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['key']['kernel']
q = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['query']['kernel']

m = np.einsum('khe,qhe->hqk', k, q)

xs_diff @ m[0] @ xs_diff.T
jax.nn.softmax(xs_diff @ m[0] @ xs_diff.T)
print(xs_diff @ m[0] @ xs_diff.T)
print('--')

# plt.hist(state.params['Dense_0']['bias'])

b = state.params['Dense_0']['bias']
b = b.reshape(-1, 1)
m[1].T @ b
print(xs_same[:1] @ m[0].T @ b)
print(xs_same[:1] @ m[0] @ b)
print('--')
print(xs_same[:1] @ m[1].T @ b)
print(xs_same[:1] @ m[1] @ b)

# <codecell>
# The two heads have vectors that are perfectly anti-aligned
a = m[0].T @ b
b = m[1].T @ b

a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

a.T @ b

# TODO: confirm parallel / antiparallel direction dictates classification
# %%
xs = feats['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['inputs'][0]

xs_same = xs[1]

xs_same @ m[1] @ xs_same.T

# <codecell>
a = np.random.randn(512, 1)
M_r = np.random.randn(512, 512)

print(np.sum(m[0] @ a))
print(np.sum(m[0].T @ a))
np.trace(m[0])