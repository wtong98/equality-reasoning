"""Simple SD task"""


# <codecell>
from pathlib import Path

from flax import traverse_util
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
from model.transformer import TransformerConfig, SimpleTransformerConfig
from task.same_different import SameDifferent 
from task.ti import TiTask

def pred_acc(n_points, a_raw=1.5):
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = n_points * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)

def pred_acc2(n_points, a_raw=1.5):
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = np.sqrt(2 * n_points) * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)


sns.set_theme(style='ticks', font_scale=1.25, rc={
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.figsize': (5.5, 4)
})

(pred_acc2(3) + 1) / 2

# pred_acc(5)

# <codecell>
df = collate_dfs('remote/6_toy_sd/data_div', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    # hist_acc = [m['accuracy'].item() for m in row['hist']['test']]
    # hist_loss = [m['loss'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        # max(hist_acc),
        # min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
adf = plot_df[plot_df['n_dims'] == 512]

mdf = adf[adf['name'].str.contains('gamma')]
mdf2 = adf[~adf['name'].str.contains('gamma')]

g = sns.lineplot(mdf, x='n_symbols', y='acc_unseen', hue='gamma0', marker='o', palette='rocket_r', alpha=0.7)
sns.lineplot(mdf2, x='n_symbols', y='acc_unseen', hue='name', marker='o', alpha=1, ax=g, palette=['C0', 'C9'], hue_order=['Adam', 'RF'])

g.legend_.set_title('')

for t in g.legend_.get_texts():
    text = t.get_text()
    if 'Adam' in text:
        t.set_text('Adam')
    elif 'RF' in text:
        t.set_text('RF')
    else:
        t.set_text(f'$\gamma_0$ = 1e{text}')

g.set_xlabel('# symbols')
g.set_ylabel('Test accuracy')
g.set_xscale('log', base=2)

g.figure.tight_layout()
sns.move_legend(g, loc='upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/cosyne/sd_acc.png ', bbox_inches='tight')

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
g.figure.savefig('fig/cosyne/sd_rich_dim.png', bbox_inches='tight')

# <codecell>
g = sns.lineplot(mdf, x='n_symbols', y='acc_unseen', hue='n_dims', marker='o', hue_norm=mpl.colors.LogNorm(), legend='full')
# pts = np.unique(mdf['n_symbols'])
# g.plot(pts, [pred_acc(p) for p in pts], 'o--')

g.set_xscale('log', base=2)


# <codecell>
mdf = adf[(adf['gamma0'] == -2)]
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
g.figure.savefig('fig/cosyne/sd_lazy_dim.png', bbox_inches='tight')

# <codecell>
### BIG K EXPERIMENT
df = collate_dfs('remote/6_toy_sd/big_k', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['train_task'].n_patches,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'n_patches', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df[plot_df['name'] == 'Adam']
mdf = mdf[(mdf['n_symbols'] == 64) | (mdf['n_symbols'] == 1024) | (mdf['n_symbols'] == 16384)]
# g = sns.lineplot(mdf, x='n_symbols', y='acc_unseen', hue='n_patches', marker='o')
gs = sns.relplot(mdf, x='n_dims', y='acc_unseen', hue='n_patches', col='n_symbols', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

gs.figure.savefig('fig/sd_adam_patchwise.png')

# <codecell>
### BIG_K SWEEP PLOTS
# mdf = plot_df[plot_df['name'] == '$\gamma_0=10^{0.0}$']
# gs = sns.relplot(mdf, x='n_dims', y='acc_seen', hue='n_patches', col='n_symbols', col_wrap=4, kind='line', marker='o')

# for g in gs.axes.ravel():
#     g.set_xscale('log', base=2)

# # <codecell>
# gs = sns.relplot(plot_df, x='n_symbols', y='acc_unseen', hue='name', col='n_dims', row='n_patches')
# for g in gs.axes.ravel():
#     g.set_xscale('log', base=2)

# plt.savefig('fig/sd_patchwise.png')


# <codecell>
### LAZY SWEEP EXPERIMENT
df = collate_dfs('remote/6_toy_sd/lazy_sweep', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_hidden', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['n_hidden'] == 256) | (mdf['n_hidden'] == 4096) | (mdf['n_hidden'] == 65536)]

gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='n_dims', col='n_hidden', kind='line', marker='o', legend='full', hue_norm=mpl.colors.LogNorm())

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

plt.savefig('fig/lazy_sweep_sel.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_hidden'] == 65536]

mdf = mdf[['n_symbols', 'n_dims', 'acc_unseen']]
mdf = mdf.groupby(['n_symbols', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_symbols', columns='n_dims', values='acc_unseen')

g = sns.heatmap(mdf)
xs = 2**np.linspace(-5, 8)
g.plot(xs, xs)

g.figure.savefig('fig/lazy_sweep_ndim_v_nsym.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_symbols'] == 16384]

mdf = mdf[['n_hidden', 'n_dims', 'acc_unseen']]
mdf = mdf.groupby(['n_hidden', 'n_dims'], as_index=False).mean()
mdf = mdf.pivot(index='n_hidden', columns='n_dims', values='acc_unseen')

g = sns.heatmap(mdf)
xs = 2**np.linspace(0, 8)
g.plot(xs, xs-1)

g.figure.savefig('fig/lazy_sweep_ndim_v_nhid.png')


# <codecell>
### NOISE SWEEP
df = collate_dfs('remote/6_toy_sd/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]
    hist_loss = [m['loss'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['sig2'],
        row['train_task'].noise,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        min(hist_loss),
    ], index=['name', 'gamma0', 'n_symbols', 'n_dims', 'sig2', 'noise', 'acc_seen', 'acc_unseen', 'acc_best', 'loss_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
p = plot_df['acc_best']
mdf = plot_df.copy()
mdf['logit'] = np.log(p / (1 - p + 1e-10))

gs = sns.relplot(mdf, x='n_dims', y='acc_best', hue='name', col='sig2', row='n_symbols', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

plt.savefig('fig/dim_by_acc.png')

# <codecell>
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
names = np.unique(mdf['name'])

for name, ax in zip(names, axs.ravel()):
    adf = mdf[mdf['name'] == name].drop('name', axis='columns')
    adf = adf.groupby(['n_symbols', 'sig2'], as_index=False)
    corrs = adf[['n_dims', 'acc_best']].cov().unstack().iloc[:,1]

    adf = adf.mean()
    adf['corrs'] = corrs

    adf = adf.pivot(index='n_symbols', columns='sig2', values='corrs')
    sns.heatmap(adf, cmap='BrBG', vmin=-20, vmax=20, ax=ax)
    ax.set_title(name)

fig.tight_layout()
# TODO: should technically be log'd then covarianced
plt.savefig('fig/acc_covariance.png')

# <codecell>
len(np.unique(mdf['name']))

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
bdf = df_bayes[df_bayes['name'] == 'Bayes Gen'].drop('name', axis='columns')
bdf = bdf.groupby(['n_symbols', 'n_dims', 'sig2']).mean()

mdf = plot_df.copy()
mdf['acc_unseen'] = mdf['acc_best']

all_bayes_acc = []
for idxs in mdf[['n_symbols', 'n_dims', 'sig2']].to_numpy():
    bayes_acc = bdf.loc[idxs[0], idxs[1], idxs[2]]['acc_unseen']
    all_bayes_acc.append(bayes_acc.item())

all_bayes_acc = np.array(all_bayes_acc)

mdf['acc_gap'] = all_bayes_acc - mdf['acc_unseen']

# <codecell>
mdf = pd.concat((mdf, df_bayes))

# mdf = mdf[mdf['sig2'] > 0.05]
gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='name', col='sig2', row='n_dims', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)
    # g.set_yscale('log')

plt.savefig('fig/noise_sweep_diff_sig_best.png')

# <codecell>
gs = sns.relplot(mdf, x='n_dims', y='acc_gap', hue='name', col='sig2', row='n_symbols', kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

plt.savefig('fig/noise_sweep_diff_sig_best_by_sig_gap.png')



# <codecell>
# NOTE: dimension dependence seems to enter when considering patch sizes > 2
n_dims = 900
n_points = 334
# n_points = np.round(0.5 * n_dims * np.log(n_dims)).astype(int)
# n_points = n_dims
# n_hidden = 892
n_hidden = 1024

noise = 0

gamma0 = 1e-5 * np.sqrt(n_dims)
# gamma0 = 1
# gamma = gamma0 * np.sqrt(n_hidden)
gamma = gamma0
lr = gamma0**2 * 0.1

n_patches = 2

train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128, noise=noise)
test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024, noise=noise)

config = MlpConfig(mup_scale=True,
                   as_rf_model=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   act_fn='relu')

# config = SimpleTransformerConfig(n_hidden=n_hidden, gamma=gamma)
# config = SimpleTransformerConfig(n_hidden=n_hidden)

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
                    loss='bce',
                    test_every=1000,
                    train_iters=5_000,
                    # lr=1e-3,
                    # optim=sign_sgd,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)


# <codecell>
xs, ys = next(test_task)
logits = state.apply_fn({'params': state.params}, xs)
np.mean((logits > 0) == ys)

np.mean(optax.sigmoid_binary_cross_entropy(logits, ys))

# <codecell>
# TODO: make plot confirming this prediction: vv
pred_acc(n_points)

# <codecell>
# W = state.params['Dense_0_freeze']['kernel']
# a = state.params['Dense_0']['kernel']

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()

plt.hist(a, bins=50)
# plt.plot(np.sort(a.flatten()))

# <codecell>
sort_idxs = np.argsort(a)

W_sort = W[:,sort_idxs]
w1, w2 = W_sort[:n_dims], W_sort[n_dims:]
dots = w1.T @ w2 / (np.linalg.norm(w1, axis=0) * np.linalg.norm(w2, axis=0))
cos_dists = np.diag(dots)

plt.plot(a[sort_idxs], cos_dists, 'o')

# <codecell>
-np.mean(a[a<0]) / np.mean(a[a>0])

# <codecell>
sort_idxs = np.argsort(a.flatten())
idx = sort_idxs[-2]

w_sel = W[:,idx].reshape(4, -1)
plt.imshow(w_sel @ w_sel.T, vmin=-5, vmax=5, cmap='bwr')
plt.colorbar()

a[idx]

# <codecell>
w_proj = w_sel @ train_task.symbols.T
plt.imshow(w_proj[:,:50], cmap='bwr')
# plt.colorbar()

# <codecell>
xs, ys = next(test_task)
out = state.apply_fn({'params': state.params}, xs)
preds = (out > 0).astype(bool)

print(np.mean(ys[ys>0] == preds[ys>0]))
print(np.mean(ys[ys==0] == preds[ys==0]))

print('---')
print(np.mean(ys[preds>0] == preds[preds>0]))
print(np.mean(ys[preds==0] == preds[preds==0]))

print('--')
print(np.mean(ys))
print(np.mean(preds))

# <codecell>
from scipy.stats import halfnorm

n_iters = 100_000
a = 1.53
L = 5

n_samps = np.round(2*L).astype(int)
# n_samps = np.round(L**2 / 2).astype(int)

s = halfnorm.rvs(0, a, size=(n_iters, n_samps))
d = halfnorm.rvs(0, 1, size=(n_iters, n_samps))

(np.mean(s.mean(axis=1) - d.mean(axis=1) > 0) + 1) / 2

# <codecell>
jax.tree.map(jnp.shape, state.params)

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel']
u, s, vh = np.linalg.svd(W)

plt.plot(s, '--o')
plt.axvline(x=n_points**2/2)
# plt.axvline(x=2*n_points-2)
plt.yscale('log')

# <codecell>
x = np.random.randn(1000, n_dims * 2) / np.sqrt(n_dims)
W_r = np.random.randn(*W.shape) / np.sqrt(n_hidden)

a_idxs = np.argsort(a.flatten())
idx = -1

out = jax.nn.relu(x @ W)
out_r = jax.nn.relu(x @ W_r)

# out = (x @ W)
# out_r = (x @ W_r)

plt.hist(  out[:,a_idxs[idx]], bins=30, alpha=0.3, density=True)
plt.hist(out_r[:,a_idxs[0]], bins=30, alpha=0.3, density=True)

print(a.flatten()[a_idxs[idx]])

xs = np.linspace(-2.5, 2.5, num=1000)
ys = 0.5 * halfnorm.pdf(xs, loc=0, scale=np.sqrt(2 * n_points / n_dims))
zs = 0.5 * halfnorm.pdf(xs, loc=0, scale=np.sqrt(2)/np.sqrt(n_hidden))

plt.plot(xs, ys, alpha=0.5)
plt.plot(xs, zs, alpha=0.5)

plt.ylim(0, 2)

# <codecell>
samp = out[5,a_idxs[:]]
plt.hist(samp[samp > 0])

# <codecell>
W_sort = np.array(W[:,a_idxs])
a_sort = np.array(a[a_idxs,:])

# W_sort[:,2:-2] = 0
# W_sort

# <codecell>
x = np.random.randn(10_000, n_dims * 2) / np.sqrt(n_dims)
out = jax.nn.relu(x @ W_sort) @ a_sort
plt.hist(out.flatten())
np.mean(out < 0)

# <codecell>
n_iters = 10_000

m = 4
a_norm = 1.5
L = 8
d = 64

# pos_res = halfnorm.rvs(size=(n_iters, m), loc=0, scale=np.sqrt(2 * L / d))
# neg_res = halfnorm.rvs(size=(n_iters, m), loc=0, scale=a_norm * np.sqrt(2 * L / d))

pos_res = halfnorm.rvs(size=(n_iters, m), loc=0, scale=1)
neg_res = halfnorm.rvs(size=(n_iters, m), loc=0, scale=a_norm)

res = pos_res.sum(axis=1) - a_norm * neg_res.sum(axis=1)
plt.hist(res)
np.mean(res < 0)

# TODO: work out distributions more carefully, try to match them as far as possible

# <codecell>
all_res = []

for curr_a in a.flatten():
    res = halfnorm.rvs(size=(n_iters), loc=0, scale=np.abs(curr_a))
    all_res.append(res)

all_res = np.array(all_res)
all_res
# <codecell>
out = all_res * a
out = out.sum(axis=0)
plt.hist(out)



# <codecell>
W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)
W_norm = np.array(W_norm)
W_sim = W_norm.T @ W_norm


# np.fill_diagonal(W_temp, 0)

dd = np.sum(W_sim, axis=1) - np.diag(W_sim)
W_sim = -W_sim
np.fill_diagonal(W_sim, dd)



# <codecell>
u, s, vh = np.linalg.svd(W_sim)
plt.plot(s[-50:], '--o')
# plt.axvline(x=2*n_points)



# <codecell>
### COORDINATE CHECKING
all_norms = []

# for n_hidden in [16, 64, 256, 1024]:
for n_patches, n_hidden in zip([2, 4, 8, 16], [16, 64, 256, 1024]):
    n_points = 8
    n_dims = 128
    # n_hidden = 512

    gamma0 = 0.01
    gamma = gamma0 * np.sqrt(n_hidden)
    # lr = gamma**2 * 0.1
    lr = gamma0**2 * 0.01

    train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
    test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

    config = MlpConfig(mup_scale=True,
                    n_out=1, 
                    vocab_size=None, 
                    n_layers=1, 
                    n_hidden=n_hidden, 
                    use_bias=False,
                    act_fn='relu')


    state, hist = train(config,
                        data_iter=iter(train_task), 
                        test_iter=iter(test_task), 
                        loss='bce',
                        gamma=gamma,
                        test_every=1000,
                        train_iters=1, 
                        optim=optax.sgd,
                        lr=lr,
                        seed=None)

    xs, _ = next(train_task)
    xs = xs.reshape(xs.shape[0], -1)

    jax.tree.map(jnp.shape, state.params)

    W = state.params['Dense_0']['kernel']
    a = state.params['Dense_1']['kernel']
    
    act = xs @ W
    norm = np.mean(np.linalg.norm(act, axis=1))
    all_norms.append(norm)

# <codecell>
plt.plot(all_norms, '--o')
# %%
### VALIDATION OF RF CALCULATIONS <-- TODO: need to complete (should be correct, 
# but need to address inconsistencies). May be issue of normalization -- pay attention to how its done exactly
n_points = 2048
n_hidden = 512
# n_dimss = np.round(np.linspace(16, 512, num=10)).astype(int)
# n_dimss = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
n_dimss = np.array([2, 4, 8, 16, 32, 64, 128])

all_avgs = []
all_avgs_n = []

for n_dims in n_dimss:
    xs_pos = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
    xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
    # xs_pos = np.sqrt(np.sqrt(2)) * np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
    xs_neg = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)

    W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)
    hp = jax.nn.relu(xs_pos @ W)

    kp = hp @ hp.T
    kp = np.triu(kp, k=1)
    # kp = kp[np.nonzero(kp)]

    hn = jax.nn.relu(xs_neg @ W)

    kn = hn @ hn.T
    kn = np.triu(kn, k=1)
    # kn = kn[np.nonzero(kn)]

    avg = np.mean(kp)

    all_avgs.append(avg)
    all_avgs_n.append(np.mean(kn))

plt.loglog(n_dimss, all_avgs, '--o')
plt.loglog(n_dimss, all_avgs_n, '--o')
# plt.loglog(n_dimss, 80 / n_dimss)
# plt.loglog(n_dimss, 80 * np.sqrt(2) / n_dimss)

# <codecell>
n_points = 1_000
n_dims = 64
n_hidden = 10_000

# task = SameDifferent(n_symbols=None, n_dims=n_dims, batch_size=n_points)
# xs, ys = next(task)
# xs = xs.reshape(xs.shape[0], -1)
# xs_pos = xs[ys==1]
# xs_neg = xs[ys==0]

xs_pos = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
# xs_pos = np.sqrt(np.sqrt(2)) * np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)
xs_neg = np.random.randn(n_points, 2*n_dims) / np.sqrt(n_dims)

# W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)
W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)

xs_pos = jax.nn.relu(xs_pos @ W)
xs_neg = jax.nn.relu(xs_neg @ W)

dp = np.triu(xs_pos @ xs_pos.T, k=1) / n_hidden
dp = dp[np.nonzero(dp)]
dn = np.triu(xs_neg @ xs_neg.T, k=1) / n_hidden
dn = dn[np.nonzero(dn)]

print(np.var(dp))
print(2*np.var(dn))

print(np.mean(dp))
print(np.mean(dn))

# print(np.mean(dp**2))
# print(np.mean(dn**2))

# dp = dp + dp**2/np.pi
# dn = dn + dn**2/np.pi

plt.hist(dp, bins=50, density=True, alpha=0.7)
plt.hist(dn, bins=50, density=True, alpha=0.7)

plt.axvline(x=np.mean(dp))
plt.axvline(x=np.mean(dn), color='red')
# plt.hist(dn * np.sqrt(2), bins=50, density=True, alpha=0.3)


# <codecell>
all_dp = []
all_dn = []

W = np.random.randn(2*n_dims, n_hidden) / np.sqrt(2*n_dims)

for _ in range(n_points):
    xs_pos = np.random.randn(2, 2*n_dims) / np.sqrt(n_dims)
    xs_pos[:,:n_dims] = xs_pos[:,n_dims:]
    xs_neg = np.random.randn(2, 2*n_dims) / np.sqrt(n_dims)

    xs_pos = jax.nn.relu(xs_pos @ W) / n_hidden
    xs_neg = jax.nn.relu(xs_neg @ W) / n_hidden

    dp = np.triu(xs_pos @ xs_pos.T, k=1)
    dp = dp[np.nonzero(dp)]
    dn = np.triu(xs_neg @ xs_neg.T, k=1)
    dn = dn[np.nonzero(dn)]

    all_dp.append(dp[0])
    all_dn.append(dn[0])


# <codecell>
plt.hist(all_dp, bins=50, alpha=0.7)
plt.hist(all_dn, bins=50, alpha=0.7)

plt.axvline(x=np.mean(all_dp))
plt.axvline(x=np.mean(all_dn), color='red')


# <codecell>
### INVESTIGATING MAX DIST
# reps = 1_000
# n = 100_000
# d = 5

# xs = np.random.randn(reps, n) / np.sqrt(d)
# maxs = np.max(xs, axis=-1)
# print(np.mean(maxs))
# print(np.sqrt(2 * np.log(n) / d))

d = 10000
n = d
xs = np.random.randn(n) / np.sqrt(d)
# print(sum(xs > 0))


xs = np.sort(xs)
# xs = xs - 3.5
plt.plot(xs)

# ss = np.linspace(0, 10_000)
# plt.loglog(ss, -1/(0.001 * (ss - 10_000)))
# plt.loglog(ss, -1/(0.001 * (ss - 10_000)))

