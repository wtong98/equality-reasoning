"""Simple SD task"""


# <codecell>
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm


import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import SameDifferent 

def pred_rich_acc(n_points, a_raw=1.5):
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = np.sqrt(2 * n_points) * np.sqrt(2 / (np.pi - 2)) * a
    neg_acc = norm.cdf(pt)
    return (neg_acc + 1) / 2

# sns.set_theme(style='ticks', font_scale=1.25, rc={
#     'axes.spines.right': False,
#     'axes.spines.top': False,
#     'figure.figsize': (5.5, 4)
# })

# <codecell>
df = collate_dfs('remote/15_sd_clean/data_div', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        # min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[
    (plot_df['n_dims'] == 512)
    & (plot_df['n_width'] == 256)
    ]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o', palette='rocket_r', alpha=0.7)
sns.lineplot(mdf2, x='n_symbols', y='acc_best', hue='name', marker='o', alpha=1, ax=g, palette=['C0', 'C9'], hue_order=['Adam', 'RF'])

xs = np.unique(mdf['n_symbols'])
acc_est = pred_rich_acc(xs, a_raw=1.5)
acc_est[0] = 0.75
plt.plot(xs, acc_est, '--o', color='red')

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    else:
        t.set_text(f'$\gamma$ = 1e{text}')

g.set_xlabel('# symbols')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
# g.figure.savefig('fig/sd_acc_sample.png', bbox_inches='tight')

# <codecell>
# mdf = plot_df[(plot_df['gamma0'] == 0) | (plot_df['gamma0'] == -2)]
adf = plot_df[plot_df['n_symbols'] >= 0] # <-- control appropriately

mdf = adf[(adf['gamma0'] == 0)]
g = sns.lineplot(mdf, x='n_dims', y='acc_unseen', hue='n_symbols', marker='o', hue_norm=mpl.colors.LogNorm(), legend='full')

# g.figure.set_size_inches(3.4, 3)
g.legend_.set_title('# symbols')


g.set_xscale('log', base=2)

g.set_xlabel('Input dimensions')
g.set_ylabel('Test accuracy')
g.set_title(r'$\gamma_0 = 1$')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/sd_rich_dim_sample.png', bbox_inches='tight')


# <codecell>
mdf = adf[(adf['gamma0'] == -4) & (adf['n_width'] == 1024)]
# mdf = plot_df[plot_df['name'] == 'RF']
g = sns.lineplot(mdf, x='n_dims', y='acc_unseen', hue='n_symbols', marker='o', hue_norm=mpl.colors.LogNorm(), legend='full')

# g.figure.set_size_inches(3.4, 3)
g.legend_.set_title('# symbols')

g.set_xscale('log', base=2)

g.set_xlabel('Input dimensions')
g.set_ylabel('Test accuracy')
g.set_title(r'$\gamma_0 \approx 0$')

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/sd_lazy_dim_sample.png', bbox_inches='tight')

# <codecell>
### LAZY VAR SYMBOLS
df = collate_dfs('remote/15_sd_clean/lazy_var_symbols', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        # min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_width'] == 4096)
    & (mdf['gamma0'] == -5)
    ]

mdf = mdf[['n_symbols', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_symbols', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_symbols', columns='n_dims', values='acc_best')

g = sns.heatmap(mdf)
xs = 2**np.linspace(-5, 8)
g.plot(xs, 1.27 * xs - 1)

g.figure.savefig('fig/lazy_sweep_ndim_v_nsym_sample.png')

# <codecell>
### LAZY VAR WIDTH
df = collate_dfs('remote/15_sd_clean/lazy_var_width', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        # min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_symbols'] == 4096)
  & (mdf['gamma0'] == -5)
    ]
 
mdf = mdf[['n_width', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_width', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_width', columns='n_dims', values='acc_best')

g = sns.heatmap(mdf)
xs = 2**np.linspace(-8, 8)
g.plot(xs, 2 * xs)

# g.figure.savefig('fig/lazy_sweep_ndim_v_nhid.png')

# <codecell>
### PHASE PORTRAIT
df = collate_dfs('remote/15_sd_clean/phase', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
acc_key = 'acc_best'
acc_cutoff = 0.9

mdf = plot_df.copy().drop('name', axis='columns')
mdf = mdf[mdf['n_symbols'] == 64].drop('n_symbols', axis='columns')

mdf = mdf.groupby(['gamma0', 'n_width', 'n_dims'], as_index=False).mean()
diff = np.array(mdf[mdf['n_dims'] == 16][acc_key]) - np.array(mdf[mdf['n_dims'] == 512][acc_key])

adf = mdf[mdf['n_dims'] == 16].drop('n_dims', axis='columns')
adf['diff'] = diff
adf.loc[adf[acc_key] < acc_cutoff, 'diff'] = np.inf

adf = adf.pivot(index='n_width', columns='gamma0', values='diff')

cmap = mpl.colormaps.get_cmap('BrBG')
cmap.set_bad('k')
sns.heatmap(adf, cmap=cmap, vmin=-0.5, vmax=0.5)


# <codecell>
### NOISE
df = collate_dfs('remote/15_sd_clean/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['config']['n_hidden'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['sig2'],
        row['train_task'].noise,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
    ], index=['name', 'gamma0', 'n_width', 'n_symbols', 'n_dims', 'sig2', 'noise', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df


# <codecell>
### COMPUTE BAYESIAN SOLUTIONS
ds = np.unique(plot_df['n_dims'])
n_symbols = np.unique(plot_df['n_symbols'])
sig2s = np.unique(plot_df['sig2'])
noise_scale = 1

n_iters = 3

all_res = []

for _, d, L, sig2 in tqdm(list(itertools.product(
                                range(n_iters), 
                                ds, 
                                n_symbols, 
                                sig2s))):

    sig2_orig = sig2

    sig2 = sig2 * noise_scale

    task = SameDifferent(n_symbols=L, n_dims=d, noise=sig2)
    test = SameDifferent(n_symbols=None, n_dims=d, batch_size=1024, noise=sig2)
    sig2 = sig2 + 1e-10

    xs, ys = next(test)

    g_preds = []
    g_logits = []
    m_preds = []
    m_logits = []

    for x in xs:
        z1, z2 = x
        g1 = log_gen_like_same(z1, z2, d, sig2)
        g2 = log_gen_like_diff(z1, z2, d, sig2)
        g_preds.append(1 if g1 > g2 else 0)
        g_logits.append(g1 - g2)

        m1 = log_mem_like_same(z1, z2, d, sig2, task.symbols)
        m2 = log_mem_like_diff(z1, z2, d, sig2, task.symbols)
        m_preds.append(1 if m1 > m2 else 0)
        m_logits.append(m1 - m2)

    all_res.extend([{
        'name': 'Bayes Gen',
        'n_symbols': L,
        'n_dims': d,
        'sig2': sig2_orig,
        'acc_unseen': np.mean(g_preds == ys),
        'loss_best': optax.sigmoid_binary_cross_entropy(np.array(g_logits), ys).mean().item()
    }, {
        'name': 'Bayes Mem',
        'n_symbols': L,
        'n_dims': d,
        'sig2': sig2_orig,
        'acc_unseen': np.mean(m_preds == ys),
        'loss_best': optax.sigmoid_binary_cross_entropy(np.array(m_logits), ys).mean().item()
    }])

df_bayes = pd.DataFrame(all_res)
df_bayes

# <codecell>
gs = sns.relplot(df_bayes, x='n_symbols', y='acc_unseen', hue='name', col='noise', row='n_dims', kind='line', marker='o')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

# <codecell>
# mdf = pd.concat((mdf, df_bayes))
mdf = plot_df.copy()
mdf = mdf[mdf['n_width'] == 1024]

# mdf = mdf[mdf['sig2'] > 0.05]
gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='name', col='sig2', row='n_dims', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)
    # g.set_yscale('log')

# plt.savefig('fig/noise_sweep_diff_sig_best.png')
