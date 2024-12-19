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
        row['info']['n_classes'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['preprocess'],
        max(hist_acc),
        hist_acc,
        np.arange(len(row['hist']['test']))
    ], index=['name', 'n_classes', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_unseen_best', 'preprocess', 'hist_acc', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.drop(['hist_acc', 'time'], axis=1).melt(id_vars=['name', 'n_classes', 'gamma0'], var_name='acc_type', value_name='acc')
mdf

gs = sns.relplot(mdf, x='n_classes', y='acc', col='acc_type', hue='name', kind='line', marker='o', palette='rocket_r')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)
   
plt.savefig('fig/cifar100_acc.png')


# <codecell>
n_hidden = 1024

train_pieces = np.arange(90)
test_pieces = np.arange(90, 100)

preprocess = True

train_task = SameDifferentCifar100(ps=train_pieces, preprocess_cnn=preprocess)
test_task = SameDifferentCifar100(ps=test_pieces, preprocess_cnn=preprocess)


gamma0 = 0.0001
gamma = gamma0
gamma = np.sqrt(n_hidden) * gamma0
lr = gamma0**2 * 1

config = MlpConfig(n_out=1, 
                   mup_scale=True,
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu',
                   as_rf_model=False
                   )


state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=10_000, 
                    seed=None,
                    optim=optax.sgd,
                    gamma=gamma,
                    lr=lr
                    )

# %%
