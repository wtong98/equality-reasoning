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
from task.same_different import SameDifferentCifar100

# <codecell>
n_hidden = 4096

train_pieces = np.arange(90)
test_pieces = np.arange(90, 100)


train_task = SameDifferentCifar100(ps=train_pieces)
test_task = SameDifferentCifar100(ps=test_pieces)


gamma0 = 1
gamma = gamma0
gamma = np.sqrt(n_hidden) * gamma0
lr = gamma0**2 * 1

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
                    seed=None,
                    optim=optax.sgd,
                    gamma=gamma,
                    lr=lr 
                    )
