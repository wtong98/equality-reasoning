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

set_theme()

# <codecell>
task = SameDifferentPsvrt(patch_size=5, n_patches=5)
xs, ys = next(task)
plt.imshow(xs[0], cmap='plasma')
plt.gca().set_axis_off()

plt.savefig('fig/psvrt_same_eg.png')

# <codecell>
df = collate_dfs('remote/5_psvrt/feature_learn')
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        len(row['train_task'].inc_set),
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        hist_acc,
        np.arange(len(row['hist']['test']))
    ], index=['name', 'n_pieces', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_unseen_best', 'hist_acc', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.drop(['hist_acc', 'time'], axis=1).melt(id_vars=['name', 'n_pieces', 'gamma0'], var_name='acc_type', value_name='acc')
mdf

gs = sns.relplot(mdf, x='n_pieces', y='acc', col='acc_type', hue='name', kind='line', marker='o')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

plt.savefig('fig/psvrt_acc.png')

# <codecell>
mdf = plot_df[plot_df['name'].str.contains('gamma')]
mdf2 = plot_df[~plot_df['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_pieces', y='acc_unseen_best', hue='gamma0', marker='o', palette='rocket_r', alpha=0.7)
sns.lineplot(mdf2, x='n_pieces', y='acc_unseen_best', hue='name', marker='o', alpha=0.7, ax=g, palette=['C0', 'C9'])

g.figure.set_size_inches(5, 4)
g.set_xscale('log', base=2)

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    else:
        t.set_text(f'$\gamma_0$ = 1e{text}')

g.set_xlabel('# shapes')
g.set_ylabel('Test accuracy')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/cosyne/psvrt_acc.png', bbox_inches='tight')


# <codecell>
mdf = plot_df.drop(['acc_seen', 'acc_unseen'], axis=1)
mdf = mdf.explode(['hist_acc', 'time'])

sns.relplot(mdf, x='time', y='hist_acc', hue='name', col='n_pieces', col_wrap=3, kind='line')
plt.savefig('fig/psvrt_hist.png')


# <codecell>
n_hidden = 512
n_train = 256
n_patches = 5
patch_size = 5

train_set = gen_patches(patch_size=patch_size, n_examples=n_train)

train_task = SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patches, inc_set=train_set)
test_task = SameDifferentPsvrt(patch_size=patch_size, n_patches=n_patches)


gamma0 = 0.1
# gamma = gamma0
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
                    optim=optax.sgd,
                    seed=None,
                    gamma=gamma,
                    lr=lr)
