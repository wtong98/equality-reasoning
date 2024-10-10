"""Visual same-different tasks"""

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
from task.same_different import SameDifferentPentomino, SameDifferentPsvrt, gen_patches
from task.pentomino import pieces

# <codecell>
n_hidden = 512
n_train = 256
n_patches = 5
patch_size = 5

train_set = gen_patches(patch_size=patch_size, n_examples=n_train)

train_task = SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patches, inc_set=train_set)
test_task = SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patches)


gamma0 = 1
# gamma = gamma0
gamma = np.sqrt(n_hidden) * gamma0
lr = gamma0**2 * 0.01

config = MlpConfig(n_out=1, 
                   mup_scale=True,
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu',
                #    as_rf_model=True
                   )


state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=100_000, 
                    optim=optax.sgd,
                    seed=None,
                    gamma=gamma,
                    lr=lr)
