"""Compare the impact of architecture on same-different performance"""


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
from model.transformer import TransformerConfig
from task.function import SameDifferent 


n_points = 128
n_dims = 256
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
                           n_mlp_layers=0,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=True)

state, hist = train(config,
                    data_iter=iter(sd_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    gamma=None,
                    test_every=1000,
                    train_iters=5_000, 
                    lr=1e-4, optim=optax.adamw,
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
jax.tree.map(np.shape, state.params)
