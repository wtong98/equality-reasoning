"""Experiments with transitive inference"""

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
n_points = 256
n_dims = 32
n_hidden = 512

n_symbols = 10

gamma = 0.01
base_lr = 0.01**2
lr = gamma * base_lr

train_task = TiTask(n_symbols=n_symbols, sep_dists=[1])
test_task = TiTask(n_symbols=n_symbols, sep_dists=[2, 3, 4])

config = MlpConfig(mup_scale=True,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   feature_learning_strength=gamma,
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
                    test_every=500,
                    train_iters=2_000, 
                    lr=lr,
                    optim=optax.sgd,
                    seed=None)

# %%
xs, ys = next(test_task)

preds = state.apply_fn({'params': state.params}, xs)
np.sign(preds) == ys
