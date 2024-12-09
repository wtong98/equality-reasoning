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
df = collate_dfs('remote/13_cifar100/feature_learn')
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        len(row['train_task'].pieces),
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        hist_acc,
        np.arange(len(row['hist']['test']))
    ], index=['name', 'n_classes', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_unseen_best', 'hist_acc', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>



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
