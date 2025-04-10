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
### PSVRT
df = collate_dfs('remote/16_vision_clean/psvrt')
df

# <codecell>
def extract_plot_vals(row):

    return pd.Series([
        row['name'],
        len(row['train_task'].inc_set),
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'n_pieces', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.melt(id_vars=['name', 'n_pieces', 'gamma0'], var_name='acc_type', value_name='acc')
mdf

gs = sns.relplot(mdf, x='n_pieces', y='acc', col='acc_type', hue='name', kind='line', marker='o')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

# plt.savefig('fig/psvrt_acc_sample.png')

# <codecell>
mdf = plot_df[plot_df['name'].str.contains('gamma')]
mdf2 = plot_df[~plot_df['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_pieces', y='acc_best', hue='gamma0', marker='o')
# sns.lineplot(mdf2, x='n_pieces', y='acc_best', hue='name', marker='o', alpha=0.7, ax=g, palette=['C0', 'C9'])

g.figure.set_size_inches(3, 2.5)

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, xmin=0, xmax=0.75, color='gray', linestyle='dashed')
g.text(x=2**9.5, y=0.49, s='chance', color='gray', fontsize=10)

g.set_xscale('log', base=2)

handles, labels = plt.gca().get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    else:
        t.set_text('$\gamma = 10^{%s}$' % int(float(text)))


g.set_xlabel('# bit-patterns')
g.set_ylabel('Test accuracy')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
# g.figure.savefig('fig/ccn/psvrt_acc_by_l.svg', bbox_inches='tight')

# <codecell>
### PSVRT
df = collate_dfs('remote/16_vision_clean/psvrt_large')
df

# <codecell>
def extract_plot_vals(row):

    return pd.Series([
        row['name'],
        len(row['train_task'].inc_set),
        row['train_task'].n_patches,
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'n_pieces', 'n_patches', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_pieces'] == 1024]

g = sns.lineplot(mdf, x='n_patches', y='acc_best', hue='gamma0', marker='o')
g.figure.set_size_inches(3.1, 2.5)

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.text(x=2, y=0.51, s='chance', color='gray', fontsize=10)

# g.set_xscale('log', base=2)

handles, labels = plt.gca().get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    else:
        t.set_text('$\gamma = 10^{%s}$' % int(float(text)))


g.set_xlabel('# patches')
g.set_ylabel('Test accuracy')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
# plt.savefig('fig/ccn/psvrt_acc_by_d.svg', bbox_inches='tight')

# <codecell>
### PENTOMINO
df = collate_dfs('remote/16_vision_clean/pentomino')
df

# <codecell>
def extract_plot_vals(row):

    return pd.Series([
        row['name'],
        len(row['train_task'].pieces),
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'n_pieces', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.melt(id_vars=['name', 'n_pieces', 'gamma0'], var_name='acc_type', value_name='acc')
mdf

gs = sns.relplot(mdf, x='n_pieces', y='acc', col='acc_type', hue='name', kind='line', marker='o')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

# plt.savefig('fig/pentomino_acc.png')

# <codecell>
mdf = plot_df[plot_df['name'].str.contains('gamma')]
mdf2 = plot_df[~plot_df['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_pieces', y='acc_best', hue='gamma0', marker='o')
# sns.lineplot(mdf2, x='n_pieces', y='acc_best', hue='name', marker='o', alpha=0.7, ax=g, palette=['C0', 'C9'])

handles, labels = plt.gca().get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.axhline(y=1, color='white', linestyle='dashed', alpha=0)
g.text(x=13, y=0.51, s='chance', color='gray', fontsize=10)

g.figure.set_size_inches(3, 2.5)
# g.set_xscale('log', base=2)

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    t.set_text('$\gamma = 10^{%s}$' % int(float(text)))

g.set_xlabel('# shapes')
g.set_ylabel('Test accuracy')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
# g.figure.savefig('fig/ccn/pentomino_acc_by_l.svg', bbox_inches='tight')

# <codecell>
### PENTOMINO LARGE
df = collate_dfs('remote/16_vision_clean/pentomino_large')
df

# <codecell>
def extract_plot_vals(row):

    return pd.Series([
        row['name'],
        len(row['train_task'].pieces),
        row['train_task'].width,
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'n_pieces', 'width', 'gamma0', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_pieces'] == 14]

g = sns.lineplot(mdf, x='width', y='acc_best', hue='gamma0', marker='o')
g.figure.set_size_inches(3, 2.5)

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, xmin=0.27, xmax=1, color='gray', linestyle='dashed')
g.axhline(y=1, color='white')
g.text(x=1.8, y=0.487, s='chance', color='gray', fontsize=10)

handles, labels = plt.gca().get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    t.set_text('$\gamma = 10^{%s}$' % int(float(text)))


ticks = np.unique(mdf['width'])[::2]
g.set_xticks(ticks)
g.set_xticklabels(ticks)

g.set_xlabel('# patches')
g.set_ylabel('Test accuracy')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
# plt.savefig('fig/ccn/pentomino_acc_by_d.svg', bbox_inches='tight')

# <codecell>
# CIFAR-100
layer_names = ['id',
               'relu1_1',
               'relu1_2',
               'relu2_1',
               'relu2_2',
               'relu3_1',
               'relu3_2',
               'relu3_3',
               'relu4_1',
               'relu4_2',
               'relu4_3',
               'relu5_1',
               'relu5_2',
               'relu5_3']

df = collate_dfs('remote/16_vision_clean/cifar100', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['n_classes'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -1,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['preprocess'],
        row['info']['actv'],
        row['info']['acc_best']
    ], index=['name', 'n_classes', 'gamma0', 'acc_seen', 'acc_unseen', 'preprocess', 'actv', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.melt(id_vars=['name', 'n_classes', 'gamma0', 'preprocess', 'actv'], var_name='acc_type', value_name='acc')

gs = sns.relplot(mdf, x='n_classes', y='acc', col='acc_type', row='actv', row_order=layer_names, hue='gamma0', kind='line', marker='o', palette='rocket_r')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)
   
plt.savefig('fig/cifar100_vgg_tmp.png')

# <codecell>
mdf = plot_df.copy()
for actv in tqdm(layer_names):
    cdf = mdf[(mdf['actv'] == actv)]
    g = sns.lineplot(cdf, x='n_classes', y='acc_best', hue='gamma0', marker='o')
    g.figure.set_size_inches(3, 2.7)

    g.legend_.set_title('')
    g.set_ylim((0.55, 0.9))

    for t in g.legend_.get_texts():
        text = t.get_text()
        t.set_text('$\gamma = 10^{%s}$' % int(float(text)))

    g.set_xlabel('# classes')
    g.set_ylabel('Test accuracy')

    g.set_xscale('log', base=2)

    g.figure.tight_layout()
    sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))

    g.set_title(actv)
    plt.savefig(f'fig/ccn/cifar100_actv/{actv}.svg', bbox_inches='tight')
    plt.show()
    # break

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['actv'] == 'relu4_1']

g = sns.lineplot(mdf, x='n_classes', y='acc_best', hue='gamma0', marker='o')
handles, labels = plt.gca().get_legend_handles_labels()
order = np.arange(len(handles))[::-1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

# g.set_ylim((0.45, 1.02))
# g.axhline(y=0.5, color='gray', linestyle='dashed')
# g.axhline(y=1, color='white', linestyle='dashed', alpha=0)

g.figure.set_size_inches(3, 2.7)
# g.set_xscale('log', base=2)

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    t.set_text('$\gamma = 10^{%s}$' % int(float(text)))

g.set_xlabel('# classes')
g.set_ylabel('Test accuracy')

g.set_xscale('log', base=2)

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('fig/ccn/cifar100_acc_by_l.svg')