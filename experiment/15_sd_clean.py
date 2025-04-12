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
    # pt = np.sqrt(2 * n_points * (1/2 * n_points - 0.5)) * np.sqrt(2 / (np.pi - 2)) * a
    prefac = 2 / (13 * (np.pi - 2))
    pt = np.sqrt(prefac * (n_points**2 - n_points))
    neg_acc = norm.cdf(pt)
    return (neg_acc + 1) / 2

set_theme()

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
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[
    (plot_df['n_dims'] == 256)
    & (plot_df['n_width'] == 1024)
    ]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o', alpha=0.7)
g.figure.set_size_inches((3.5, 2.7))
# sns.lineplot(mdf2, x='n_symbols', y='acc_best', hue='name', marker='o', alpha=1, ax=g, palette=['C0', 'C9'], hue_order=['Adam', 'RF'])

xs = np.unique(mdf['n_symbols'])
acc_est = pred_rich_acc(xs, a_raw=1.5)
acc_est[0] = 0.75

tdf = pd.DataFrame({'n_symbols': xs, 'acc_best': acc_est})
tdf['name'] = 'Theory'
sns.lineplot(tdf, x='n_symbols', y='acc_best', hue='name', ax=g, palette=['red'], linestyle='dashed')

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.text(2**8.5, 0.51, 'chance', color='gray', fontsize=10)

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [6, 5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    elif text != 'Theory':
        val = np.round(10**float(text), decimals=2)
        if val != 0:
            t.set_text(f'$\gamma = {val:.2f}$')
        else:
            t.set_text(r'$\gamma \approx 0.00$')

g.set_xlabel('# symbols ($L$)')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)


g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/ccn/sd_by_l.svg', bbox_inches='tight')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_symbols'] == 16)
    & (mdf['n_width'] == 1024)
    & mdf['name'].str.contains('gamma')
    ]

g = sns.lineplot(mdf, x='n_dims', y='acc_best', hue='gamma0', marker='o')
g.figure.set_size_inches((3.5, 2.7))

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')
g.text(2**4, 0.51, 'chance', color='gray', fontsize=10)

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    val = np.round(10**float(text), decimals=2)
    if val != 0:
        t.set_text(f'$\gamma = {val:.2f}$')
    else:
        t.set_text(r'$\gamma \approx 0.00$')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/ccn/sd_by_d.svg', bbox_inches='tight')

# <codecell>
# mdf = plot_df[(plot_df['gamma0'] == 0) | (plot_df['gamma0'] == -2)]
adf = plot_df[plot_df['n_symbols'] >= 0] # <-- control appropriately

mdf = adf[(adf['gamma0'] == 0)]
g = sns.lineplot(mdf, x='n_dims', y='acc_unseen', hue='n_symbols', marker='o', hue_norm=mpl.colors.LogNorm(), legend='full')

g.figure.set_size_inches(4, 3.5)
g.legend_.set_title('# symbols')

g.set_xscale('log', base=2)

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('Test accuracy')
g.set_title(r'$\gamma = 1$')

g.set_ylim((0.45, 1.02))
g.axhline(y=0.5, color='gray', linestyle='dashed')

g.figure.tight_layout()

sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/ccn/sd_rich_dim.svg', bbox_inches='tight')

# <codecell>
mdf = adf[(adf['gamma0'] == -4) & (adf['n_width'] == 1024)]
# mdf = plot_df[plot_df['name'] == 'RF']
g = sns.lineplot(mdf, x='n_dims', y='acc_unseen', hue='n_symbols', marker='o', hue_norm=mpl.colors.LogNorm(), legend='full')

g.set_ylim((0.45, 1.02))

g.figure.set_size_inches(4, 3.5)
g.legend_.set_title('# symbols')

g.set_xscale('log', base=2)

g.set_xlabel('Input dimension (d)')
g.set_ylabel('Test accuracy')
g.set_title(r'$\gamma_0 \approx 0$')

g.axhline(y=0.5, color='gray', linestyle='dashed')

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/ccn/sd_lazy_dim.svg', bbox_inches='tight')

# <codecell>
### LAZY VAR SYMBOLS
df = collate_dfs('remote/15_sd_clean/lazy_var_symbols', concat=True)
# df = collate_dfs('remote/15_sd_clean/lazy_test', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
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

mdf = mdf.iloc[::-1]
g = sns.heatmap(mdf, square=False)
g.figure.set_size_inches(3.5, 2.7)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 20 - 2 * xs + 0.5, color='black', linestyle='dashed')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('# symbols ($L$)')

g.figure.savefig('fig/ccn/lazy_ndim_v_nsym.svg')
# g.figure.savefig('fig/lazy_ndim_v_nsym.png', bbox_inches='tight')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_width'] == 4096)
    & (mdf['gamma0'] == 0)
    ]

mdf = mdf[['n_symbols', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_symbols', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_symbols', columns='n_dims', values='acc_best')

mdf = mdf.iloc[::-1]
g = sns.heatmap(mdf, vmin=0.5, vmax=1)
g.figure.set_size_inches(3.5, 2.7)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 16.5 - 0 * xs, color='black', linestyle='dashed')

g.set_xlabel('Input dimension ($d$)')
g.set_ylabel('# symbols ($L$)')

# g.figure.savefig('fig/ccn/rich_ndim_v_nsym.svg')
g.figure.savefig('fig/rich_ndim_v_nsym.png', bbox_inches='tight')

# <codecell>
### LAZY VAR WIDTH
df = collate_dfs('remote/15_sd_clean/lazy_var_width', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_symbols'] == 128)
  & (mdf['gamma0'] == -5)
    ]
 
mdf = mdf[['n_width', 'n_dims', 'acc_best']]
mdf = mdf.groupby(['n_width', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_width', columns='n_dims', values='acc_best')

g = sns.heatmap(mdf)
xs = 2**np.linspace(-8, 8)
# g.plot(xs, xs + np.log(xs))
# g.plot(xs, xs)

# g.figure.savefig('fig/lazy_sweep_ndim_v_nhid_sample.png')

# <codecell>
### PHASE PORTRAIT
df = collate_dfs('remote/15_sd_clean/phase', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
acc_key = 'acc_best'
acc_cutoff = 0.9

mdf = plot_df.copy().drop('name', axis='columns')
mdf = mdf[mdf['n_symbols'] == 64].drop('n_symbols', axis='columns')
mdf['gamma0'] = np.round(mdf['gamma0'], decimals=2)
mdf

mdf = mdf.groupby(['gamma0', 'n_width', 'n_dims'], as_index=False).mean()
diff = np.array(mdf[mdf['n_dims'] == 16][acc_key]) - np.array(mdf[mdf['n_dims'] == 512][acc_key])

adf = mdf[mdf['n_dims'] == 16].drop('n_dims', axis='columns')
adf['diff'] = diff
adf.loc[adf[acc_key] < acc_cutoff, 'diff'] = np.inf

adf = adf.pivot(index='n_width', columns='gamma0', values='diff')
adf = adf.iloc[::-1]

cmap = mpl.colormaps.get_cmap('BrBG')
cmap.set_bad('k')
sns.heatmap(adf, cmap=cmap, vmin=-0.5, vmax=0.5)

# plt.savefig('fig/phase_sample.png')

# <codecell>
### NOISE
df = collate_dfs('remote/15_sd_clean/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
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
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_width', 'n_symbols', 'n_dims', 'sig2', 'noise', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df


# <codecell>
df_bayes = collate_dfs('remote/15_sd_clean/bayes', concat=True)
df_bayes

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
gs = sns.relplot(df_bayes, x='n_symbols', y='acc_unseen', hue='name', col='sig2', row='n_dims', kind='line', marker='o')
for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

# <codecell>
mdf = plot_df.copy()

mdf = mdf[(mdf['n_width'] == 1024)]
# sigs = [0, 0.1, 1, 2, 4]
sigs = [0, 1, 2, 4]
# all_n_dims = [64, 128, 256]
all_n_dims = [64]

for sig, n_dims in tqdm(list(itertools.product(sigs, all_n_dims))):
    cdf = mdf[(mdf['noise'] == sig) & (mdf['n_dims'] == n_dims)]
    bdf = df_bayes[(df_bayes['sig2'] == sig) & (df_bayes['n_dims'] == n_dims)]

    g = sns.lineplot(cdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o')
    sns.lineplot(bdf, x='n_symbols', y='acc_unseen', hue='name', ax=g, linestyle='dashed', errorbar=('ci', False), palette=['red', 'magenta'])

    g.set_ylim((0.45, 1.02))
    # g.axhline(y=0.5, color='gray', linestyle='dashed')
    g.axhline(y=1, color='white', linestyle='dashed', alpha=0)

    g.set_xscale('log', base=2)
    g.set_xlabel('# symbols ($L$)')
    g.set_ylabel('Test accuracy')
    g.set_title(f'$\sigma^2 = {sig}$')

    g.legend_.set_title('')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [5, 4, 3, 2, 1, 0, 6, 7]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    for t in g.legend_.get_texts():
        text = t.get_text()
        if 'Gen' in text:
            t.set_text('Bayes gen')
        elif 'Mem' in text:
            t.set_text('Bayes mem')
        elif text != 'Theory':
            if float(text[-1]) != 5:
                t.set_text('$\gamma$ = $10^{%s}$' % text)
            else:
                t.set_text(r'$\gamma \approx 0$')

    g.figure.set_size_inches(3, 2.6)
    g.figure.tight_layout()
    sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(f'fig/ccn/bayes/d_{n_dims}_sig2_{sig}.svg', bbox_inches='tight')
    plt.show()

# <codecell>
mdf = pd.concat((plot_df, df_bayes))

# mdf = mdf[mdf['sig2'] > 0.05]
# gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='gamma0', col='sig2', row='n_dims', kind='line', marker='o', palette='rocket_r')
gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='name', col='sig2', row='n_dims', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)
    # g.set_yscale('log')

plt.savefig('fig/noise_sweep_diff_sig_best_sample.png')


# <codecell>
df = collate_dfs('remote/15_sd_clean/rich', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['config']['n_hidden'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['info']['acc_best']
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_width', 'acc_seen', 'acc_unseen', 'acc_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[
    (plot_df['n_dims'] == 256)
    & (plot_df['n_width'] == 1024)
    ]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_best', hue='gamma0', marker='o', alpha=0.7)
g.figure.set_size_inches((3, 2.4))
# sns.lineplot(mdf2, x='n_symbols', y='acc_best', hue='name', marker='o', alpha=1, ax=g, palette=['C0', 'C9'], hue_order=['Adam', 'RF'])

xs = np.unique(mdf['n_symbols'])
acc_est = pred_rich_acc(xs, a_raw=1.5)
acc_est[0] = 0.75

tdf = pd.DataFrame({'n_symbols': xs, 'acc_best': acc_est})
tdf['name'] = 'Theory'
sns.lineplot(tdf, x='n_symbols', y='acc_best', hue='name', ax=g, palette=['red'], linestyle='dashed')

# g.set_ylim((0.45, 1.02))
# g.axhline(y=0.5, color='gray', linestyle='dashed')

g.legend_.set_title('')

handles, labels = plt.gca().get_legend_handles_labels()
order = [6, 5, 4, 3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    elif text != 'Theory':
        t.set_text('$\gamma$ = $10^{%s}$' % np.round(float(text), decimals=1))

g.set_xlabel('# symbols ($L$)')
g.set_ylabel('Test accuracy')
# g.set_xscale('log', base=2)


g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/ccn/rich.svg', bbox_inches='tight')


# <codecell>
n_iters = 1000
n_dims = 10000
a = np.random.randn(n_iters, n_dims)
b = np.random.randn(n_iters, n_dims)
c = np.random.randn(n_iters, n_dims)
d = np.random.randn(n_iters, n_dims)
e = np.random.randn(n_iters, n_dims)
f = np.random.randn(n_iters, n_dims)
g = np.random.randn(n_iters, n_dims)
h = np.random.randn(n_iters, n_dims)

z1 = np.diag(a @ (b + c).T)
z2 = np.diag(d @ e.T + f @ g.T)

plt.hist(z1, bins=50, density=True, alpha=0.5)
plt.hist(z2, bins=50, density=True, alpha=0.5)

# <codecell>
n_reps = 1000

n_dims = 100

xs = np.random.randn(n_reps, n_dims) / np.sqrt(n_dims)
ys = np.random.randn(n_reps, n_dims) / np.sqrt(n_dims)

u = np.abs(xs.sum(axis=-1) + ys.sum(axis=-1))
v = np.abs(xs.sum(axis=-1) - ys.sum(axis=-1))

rho = 10
print(np.mean(rho * v - u > 0))
print(2 * np.arctan(rho) / np.pi)

# <codecell>
