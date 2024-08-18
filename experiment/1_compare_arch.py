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
from task.function import SameDifferent 

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
n_points = 64
n_dims = 128
n_hidden = 512

sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

gamma0 = 1000
lr = gamma0 * 0.1
# lr = optax.exponential_decay(1, transition_steps=2000, decay_rate=2_000_000_000)
# lr = 1

# config = MlpConfig(mup_scale=False,
#                    n_out=1, 
#                    vocab_size=None, 
#                    n_layers=1, 
#                    n_hidden=n_hidden, 
#                    feature_learning_strength=gamma0,
#                    use_bias=False,
#                    act_fn='relu')

config = TransformerConfig(n_layers=1,
                           n_hidden=512,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=2,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=False)

state, hist = train(config,
                    data_iter=iter(sd_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    gamma=None,
                    test_every=1000,
                    train_iters=50_000, 
                    # lr=1e-2, optim=optax.sgd,
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


# %%
xs_same = np.random.randn(2, n_dims) / np.sqrt(n_dims)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, n_dims) / np.sqrt(n_dims)

xs = np.stack([xs_same, xs_diff])

out, feats = state.apply_fn({'params': state.params}, xs, mutable='intermediates')
# out, feats = state.apply_fn({'params': state.params}, xs, capture_intermediates=True)
# feats

att_same, att_diff = list(traverse_util.flatten_dict(feats).values())[0][0]
print(att_same)
print(att_diff)
print(out)

print(jax.tree.map(np.shape, state.params))
val_mat = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
proj_mat = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()

in_mat = state.params['Dense_0']['kernel']
out_mat = state.params['Dense_1']['kernel']

xs_diff @ in_mat @ val_mat @ proj_mat @ out_mat



# %%
