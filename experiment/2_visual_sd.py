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
from task.same_different import SameDifferentPentomino
from task.pentomino import pieces

# <codecell>
df = collate_dfs('remote/2_visual_same_diff/feature_learn')
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        len(row['train_task'].pieces),
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        hist_acc,
        np.arange(len(row['hist']['test']))
    ], index=['name', 'n_pieces', 'acc_seen', 'acc_unseen', 'acc_unseen_best', 'hist_acc', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.drop(['hist_acc', 'time'], axis=1).melt(id_vars=['name', 'n_pieces'], var_name='acc_type', value_name='acc')
mdf

sns.relplot(mdf, x='n_pieces', y='acc', col='acc_type', hue='name', kind='line', marker='o')
plt.savefig('fig/visual_sd_smooth.png')

# <codecell>
mdf = plot_df.drop(['acc_seen', 'acc_unseen'], axis=1)
mdf = mdf.explode(['hist_acc', 'time'])

sns.relplot(mdf, x='time', y='hist_acc', hue='name', col='n_pieces', col_wrap=3, kind='line')
plt.savefig('fig/visual_sd_acc_curves_smooth.png')


# <codecell>
### LARGE PLOT
df = collate_dfs('remote/2_visual_same_diff/large')
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['train_task'].width,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        hist_acc,
        np.arange(len(row['hist']['test']))
    ], index=['name', 'n_width', 'acc_seen', 'acc_unseen', 'acc_unseen_best', 'hist_acc', 'time'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.drop(['hist_acc', 'time'], axis=1).melt(id_vars=['name', 'n_width'], var_name='acc_type', value_name='acc')
mdf

sns.relplot(mdf, x='n_width', y='acc', col='acc_type', hue='name', kind='line', marker='o')
plt.savefig('fig/visual_sd_width_wise.png')

# <codecell>
mdf = plot_df.drop(['acc_seen', 'acc_unseen'], axis=1)
mdf = mdf.explode(['hist_acc', 'time'])

sns.relplot(mdf, x='time', y='hist_acc', hue='name', col='n_width', col_wrap=3, kind='line')
plt.savefig('fig/visual_sd_width_acc_curves.png')

# <codecell>
threshold = 0.9

def extract(row):
    time_idx = np.argmax(np.array(row['hist_acc']) > threshold)

    return pd.Series([
        row['name'],
        row['n_width'],
        time_idx
    ], index=['name', 'n_width', 'time'])

mdf = plot_df.apply(extract, axis=1).reset_index(drop=True)

# <codecell>
g = sns.lineplot(mdf, x='n_width', y='time', marker='o', hue='name')

widths = np.arange(2, 9)
time_est = widths
g.plot(widths, 6 * time_est, '--', color='black')
g.plot(widths, time_est**2, '--', color='black')
g.plot(widths, 0.5 * time_est**2 * np.log(time_est), '--', color='black')

g.set_xscale('log')
g.set_yscale('log')

plt.savefig('fig/visual_sd_width_success_thresh.png')

# <codecell>
n_hidden = 512

width = 2
ps = np.random.permutation(np.arange(18))
n_train = 16

ps_train = ps[:n_train]
ps_test = ps[n_train:]

train_task = SameDifferentPentomino(ps=ps_train, width=width, batch_size=128, blur=0, random_blur=True)
test_task = SameDifferentPentomino(ps=ps_test, width=width, batch_size=128, blur=0)

gamma0 = 10
lr = gamma0 * 0.1

config = MlpConfig(n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                  #  use_bias=False,
                   act_fn='relu',
                   feature_learning_strength=gamma0,
                #    as_rf_model=True
                   )


# config = TransformerConfig(n_layers=1,
#                            n_hidden=513,
#                            pos_emb=False,
#                            n_mlp_layers=0,
#                            n_heads=3,
#                            layer_norm=False,
#                            as_rf_model=False,
#                            residual_connections=False,
#                            use_simple_att=True,
#                            freeze_emb=False)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=10_000, 
                    optim=optax.sgd,  # NOTE: sharp contrast in using adam vs sgd
                    seed=None,
                    # lr=1e-4
                    lr=lr
                    # lr=optax.schedules.exponential_decay(1e-2, 
                    #                                      transition_steps=30_000,
                    #                                      decay_rate=0.9,
                    #                                      transition_begin=5_000,
                    #                                      end_value=1e-6)
                                                         )

'''
Observations:
- Exhibits "grokking"-like phenomenon in recognizing other shapes
- Attains full success eventually
- Probably a feature learning story here
- SignSGD does't seem to work as well (may need more tuning / momentum?)
'''

# %%
jax.tree.map(np.shape, state.params)

w = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel']

# idx = 462
# plt.imshow(w[:,idx].reshape(14, 14))
# a[idx]

all_pieces = []
for p in pieces:
    all_pieces.extend(p)


sort_idxs = np.argsort(a.flatten())
im_dir = summon_dir('fig/visual_sd_comps', clear_if_exists=True)

piece_idxs = [len(p) for p in pieces[:-1]]
piece_idxs = np.cumsum(piece_idxs)

for rank, idx in tqdm(enumerate(sort_idxs), total=len(sort_idxs)):
    zs = np.array(all_pieces)
    zs = zs.reshape(zs.shape[0], -1)

    w_orig = w[:,idx].reshape(14, 14)
    plt.imshow(w_orig)
    plt.savefig(im_dir / f'{rank}_w_orig.png')
    plt.clf()

    w_reco = np.zeros(w_orig.shape)
    coords = [1, 8]
    fig, axs = plt.subplots(2, 2, figsize=(20, 6))
    for x in coords:
        for y in coords:
            w_sel = w_orig[x:x+5,y:y+5]
            w_sel = w_sel.reshape(-1, 1)

            betas = np.linalg.pinv(zs @ zs.T) @ zs @ w_sel
            pred = zs.T @ betas
            pred = pred.reshape(5, 5)

            w_reco[x:x+5,y:y+5] = pred

            ax_x = 0 if x == 1 else 1
            ax_y = 0 if y == 1 else 1
            ax = axs[ax_x, ax_y]

            ax.plot(betas, 'o')
            for p in piece_idxs:
                ax.axvline(x=p-0.5, color='red', alpha=0.2)

            ax.axhline(y=0, color='gray', linestyle='dashed', alpha=0.5)
            ax.set_xlabel('piece idx')
            ax.set_ylabel('coefficient')
    
    fig.tight_layout()
    fig.savefig(im_dir / f'{rank}_betas.png')
    
    plt.clf()

    plt.imshow(w_reco)
    plt.savefig(im_dir / f'{rank}_w_reco.png')

# <codecell>
plt.plot(a.flatten()[sort_idxs])
plt.savefig(im_dir / 'a.png')

# %%
from task.pentomino import pieces

all_pieces = []
for p in pieces:
    all_pieces.extend(p)

corrs = np.zeros((len(all_pieces),)*2)
for i, pi in enumerate(all_pieces):
    for j, pj in enumerate(all_pieces):
        corrs[i,j] = np.sum(pi * pj)


plt.imshow(corrs)
plt.colorbar()