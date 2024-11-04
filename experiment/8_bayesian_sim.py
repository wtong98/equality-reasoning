"""Confirming Bayesian description with simulation"""

# <codecell>
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
import seaborn as sns

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import * 

def gauss(x, means, var):
    cov = np.identity(d) * var
    vals = [multivariate_normal.pdf(
        x, mean=m, cov=cov) for m in means]
    return np.array(vals)

def log_gauss(x, means, var):
    cov = np.identity(d) * var
    vals = [multivariate_normal.logpdf(
        x, mean=m, cov=cov) for m in means]
    return np.array(vals)

# <codecell>
sig2 = 2
d = 2

### SAME
s1_true = np.random.randn(d) / np.sqrt(d)
z1 = np.random.randn(d) * (np.sqrt(sig2/d)) + s1_true
z2 = np.random.randn(d) * (np.sqrt(sig2/d)) + s1_true

# <codecell>
n_iters = 10_000

s1 = np.random.randn(n_iters, d) / np.sqrt(d)

t1 = gauss(z1, s1, sig2/d)
t2 = gauss(z2, s1, sig2/d)

pos_like = np.mean(t1 * t2)
pos_like


# <codecell>
def gen_like_same(z1, z2, d, sig2):
    t1 = (2 * np.pi / d)**(-d)
    t2 = sig2 ** (-d/2)
    t3 = (1 / (2 + sig2))**(d/2)

    a1 = (1 + sig2) / (2 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = (2 / (2 + sig2))
    a4 = np.dot(z1, z2)

    t4 = - d/(2 * sig2)
    t5 = a1 * a2 - a3 * a4

    t_exp = np.exp(t4 * t5)
    sol = t1 * t2 * t3 * t_exp
    return sol

def log_gen_like_same(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = (-d/2) * np.log(sig2)
    t3 = (-d / 2) * np.log(2 + sig2)

    a1 = (1 + sig2) / (2 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = (2 / (2 + sig2))
    a4 = np.dot(z1, z2)

    t4 = - d/(2 * sig2)
    t5 = a1 * a2 - a3 * a4

    t_exp = t4 * t5
    log_sol = t1 + t2 + t3 + t_exp
    return log_sol

# gen_like_same(z1, z2, d, sig2)
# np.exp(log_gen_like_same(z1, z2, d, sig2))

# <codecell>
sig2 = 2
d = 3

def gauss(x, means, var):
    cov = np.identity(d) * var
    vals = [multivariate_normal.pdf(
        x, mean=m, cov=cov) for m in means]
    return np.array(vals)

### DIFFERENT
s1_true = np.random.randn(d) / np.sqrt(d)
s2_true = np.random.randn(d) / np.sqrt(d)
z1 = np.random.randn(d) * (np.sqrt(sig2/d)) + s1_true
z2 = np.random.randn(d) * (np.sqrt(sig2/d)) + s2_true

# <codecell>
n_iters = 10_000

s1 = np.random.randn(n_iters, d) / np.sqrt(d)
s2 = np.random.randn(n_iters, d) / np.sqrt(d)

t1 = gauss(z1, s1, sig2/d)
t2 = gauss(z2, s2, sig2/d)

pos_diff = np.mean(t1 * t2)
pos_diff

# <codecell>
def gen_like_diff(z1, z2, d, sig2):
    t1 = (2 * np.pi / d)**(-d)
    t2 = (1 / (1 + sig2))**(d)

    a1 = 1 / (1 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)

    t4 = - d/2
    t5 = a1 * a2

    t_exp = np.exp(t4 * t5)
    sol = t1 * t2 * t_exp
    return sol

def log_gen_like_diff(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = -d * np.log(1 + sig2)

    a1 = 1 / (1 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)

    t4 = - d/2
    t5 = a1 * a2

    t_exp = t4 * t5
    log_sol = t1 + t2 + t_exp
    return log_sol

# gen_like_diff(z1, z2, d, sig2)
# np.exp(log_gen_like_diff(z1, z2, d, sig2))


# <codecell>
# ss = np.random.randn(10000, d) / np.sqrt(d)
def mem_like_same(z1, z2, d, sig2, ss):
    t1 = (d / (2 * np.pi * sig2))**d

    a1 = (-d / (2 * sig2))
    a2 = (1/2) * (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = np.dot(z1, z2)

    p1 = t1 * np.exp(a1 * (a2 - a3))

    d1 = -d / sig2
    d2 = ss - (z1 + z2) / 2
    d_exp = np.exp(d1 * np.linalg.norm(d2, axis=1)**2)

    sol = p1 * np.mean(d_exp)
    return sol
    

def log_mem_like_same(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = (-d / (2 * sig2))
    a2 = (1/2) * (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = np.dot(z1, z2)

    p1 = t1 + a1 * (a2 - a3)

    d1 = -d / sig2
    d2 = ss - (z1 + z2) / 2
    d_exp = d1 * np.linalg.norm(d2, axis=1)**2

    log_sol = p1 + logsumexp(d_exp) - np.log(L)
    return log_sol

# print(mem_like_same(z1, z2, d, sig2, ss))
# print(np.exp(log_mem_like_same(z1, z2, d, sig2, ss)))
# gen_like_same(z1, z2, d, sig2)

# <codecell>
# mem like diff
# L = 5
# ss = np.random.randn(L, d) / np.sqrt(d)

def mem_like_diff(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = (d / (2 * np.pi * sig2))**d

    a1 = -d / (2 * sig2)
    e1s = np.exp(a1 * np.linalg.norm(ss - z1, axis=1)**2)
    e2s = np.exp(a1 * np.linalg.norm(ss - z2, axis=1)**2)

    prods = np.outer(e1s, e2s)
    np.fill_diagonal(prods, 0)

    t2 = 1 / (L * (L - 1))
    sol = t1 * t2 * np.sum(prods)
    return sol


def log_mem_like_diff(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = -d / (2 * sig2)
    e1s = np.reshape(a1 * np.linalg.norm(ss - z1, axis=1)**2, (-1, 1))
    e2s = np.reshape(a1 * np.linalg.norm(ss - z2, axis=1)**2, (1, -1))

    prods = e1s + e2s
    np.fill_diagonal(prods, -np.inf)

    t2 = -np.log(L * (L - 1))
    log_sol = t1 + t2 + logsumexp(prods)
    return log_sol

# print(mem_like_diff(z1, z2, d, sig2, ss))
# print(np.exp(log_mem_like_diff(z1, z2, d, sig2, ss)))
# gen_like_diff(z1, z2, d, sig2)


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
ds = np.unique(plot_df['n_dims'])
n_symbols = np.unique(plot_df['n_symbols'])[:-2]

sig2 = 1e-10

all_res = []

for d, L in tqdm(list(itertools.product(ds, n_symbols))):
    task = SameDifferent(n_symbols=L, n_dims=d)
    test = SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)

    xs, ys = next(test)

    g_preds = []
    m_preds = []

    for x in xs:
        z1, z2 = x
        g1 = log_gen_like_same(z1, z2, d, sig2)
        g2 = log_gen_like_diff(z1, z2, d, sig2)
        g_preds.append(1 if g1 > g2 else 0)
        # print('g1', g1)
        # print('g2', g2)

        m1 = log_mem_like_same(z1, z2, d, sig2, task.symbols)
        m2 = log_mem_like_diff(z1, z2, d, sig2, task.symbols)
        # print('m1', m1)
        # print('m2', m2)
        # print('---')
        m_preds.append(1 if m1 > m2 else 0)


    # g_preds = np.array(g_preds)
    # print(np.mean(g_preds == ys))

    # m_preds = np.array(m_preds)
    # print(np.mean(m_preds == ys))

    all_res.extend([{
        'name': 'Bayes Gen',
        'n_symbols': L,
        'n_dims': d,
        'acc_unseen': np.mean(g_preds == ys)
    }, {
        'name': 'Bayes Mem',
        'n_symbols': L,
        'n_dims': d,
        'acc_unseen': np.mean(m_preds == ys)
    }])

df_bayes = pd.DataFrame(all_res)
df_bayes

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['name'] == 'Adam') | (mdf['name'] == 'RF')]
mdf = pd.concat((mdf, df_bayes))

gs = sns.relplot(mdf, x='n_symbols', y='acc_unseen', hue='name', col='n_dims', col_wrap=4, kind='line', marker='o')

for g in gs.axes.ravel():
    g.set_xscale('log', base=2)

plt.savefig('fig/bayes_gen_mem.png')

# %%
### LAZY GEN MODEL
d = 2
sig2 = 1e-5

s1_true = np.random.randn(d) / np.sqrt(d)
s2_true = np.random.randn(d) / np.sqrt(d)

z1 = np.random.randn(d) * (np.sqrt(sig2/d)) + s1_true
z2 = np.random.randn(d) * (np.sqrt(sig2/d)) + s2_true

n_us = 100
temp = 10

s1 = z1[None,...]
s2 = z2[None,...]

u1s = np.random.randn(n_us, d) / np.sqrt(d)
u2s = np.random.randn(n_us, d) / np.sqrt(d)

dists1 = u1s[None,...] - s1[:,None,:]
dists2 = u2s[None,...] - s2[:,None,:]
raw_scores = np.linalg.norm(dists1, axis=-1)**2 + np.linalg.norm(dists2, axis=-1)**2

# s1_norm = s1 / np.linalg.norm(s1, axis=1, keepdims=True)
# s2_norm = s2 / np.linalg.norm(s2, axis=1, keepdims=True)

# u1s_norm = u1s / np.linalg.norm(u1s, axis=1, keepdims=True)
# u2s_norm = u2s / np.linalg.norm(u2s, axis=1, keepdims=True)

# raw_scores = temp * (s1_norm @ u1s_norm.T + s2_norm @ u2s_norm.T)
raw_scores = raw_scores - np.mean(raw_scores, axis=1, keepdims=True)
probs = np.exp(-temp * raw_scores)
probs = probs / np.sum(probs, axis=1, keepdims=True)

u1_mean = np.sum(u1s[None,...] * probs[...,None], axis=1)
u2_mean = np.sum(u2s[None,...] * probs[...,None], axis=1)

print(s1)
print(s2)
print(u1_mean)
print(u2_mean)

print(np.linalg.norm(u1_mean))
print(np.linalg.norm(u2_mean))

print(logsumexp(log_gauss(z1, u1_mean, sig2)))
print(logsumexp(log_gauss(z2, u2_mean, sig2)))

# <codecell>
def sim_lazy(z1, z2, n_iters, n_us, relation, d=2, temp=10, sig2=0.01):
    s1 = np.random.randn(n_iters, d) / np.sqrt(d)
    s2 = None

    # begin cheating
    s1 = z1[None,...]
    s2 = z2[None,...]
    # end cheating

    if relation == 'same':
        if np.linalg.norm(z1 - z2)**2 > 0.1: # cheat condition
            s2 = s1
    elif relation == 'diff':
        # s2 = np.random.randn(n_iters, d) / np.sqrt(d)
        if np.linalg.norm(z1 - z2)**2 < 0.1: # cheat condition
            s2 = np.random.randn(1, d) / np.sqrt(d)
    else:
        raise ValueError(f'unrecognized relation: {relation}')


    u1s = np.random.randn(n_us, d) / np.sqrt(d)
    u2s = np.random.randn(n_us, d) / np.sqrt(d)

    dists1 = u1s[None,...] - s1[:,None,:]
    dists2 = u2s[None,...] - s2[:,None,:]
    raw_scores = np.linalg.norm(dists1, axis=-1)**2 + np.linalg.norm(dists2, axis=-1)**2

    # s1_norm = s1 / np.linalg.norm(s1, axis=1, keepdims=True)
    # s2_norm = s2 / np.linalg.norm(s2, axis=1, keepdims=True)

    # u1s_norm = u1s / np.linalg.norm(u1s, axis=1, keepdims=True)
    # u2s_norm = u2s / np.linalg.norm(u2s, axis=1, keepdims=True)

    # raw_scores = temp * (s1_norm @ u1s_norm.T + s2_norm @ u2s_norm.T)
    raw_scores = raw_scores - np.mean(raw_scores, axis=1, keepdims=True)
    probs = np.exp(-temp * raw_scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    u1_mean = np.sum(u1s[None,...] * probs[...,None], axis=1)
    u2_mean = np.sum(u2s[None,...] * probs[...,None], axis=1)
    # print('u1_mean', u1_mean)
    # print('u2_mean', u2_mean)

    # print(s1)
    # print(s2)
    # print(u1_mean)
    # print(u2_mean)

    ll = logsumexp(log_gauss(z1, u1_mean, sig2)) \
        + logsumexp(log_gauss(z2, u2_mean, sig2)) \
        - np.log(n_iters)

    return ll

print(sim_lazy(z1, z2, 10_000, 1024, 'same', d=2, temp=10, sig2=1e-5))
print(sim_lazy(z1, z2, 10_000, 1024, 'diff', d=2, temp=10, sig2=1e-5))

# <codecell>
df = collate_dfs('remote/6_toy_sd/lazy_sweep', concat=True)
df

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
mdf = mdf[mdf['n_symbols'] == 16384]
ds = np.unique(mdf['n_dims'])
n_hidden = np.unique(mdf['n_hidden'])[:-2]

sig2 = 1e-10

all_res = []

for d, n_us in tqdm(list(itertools.product(ds, n_hidden))):
    n_iters = 10_00
    temp = 15
    sig2 = 1e-5

    test = SameDifferent(n_symbols=None, n_dims=d, batch_size=500)

    xs, ys = next(test)

    g_preds = []
    m_preds = []

    for x in xs:
        z1, z2 = x
        g1 = sim_lazy(z1, z2, n_iters, n_us, relation='same', d=d, temp=temp, sig2=sig2)
        g2 = sim_lazy(z1, z2, n_iters, n_us, relation='diff', d=d, temp=temp, sig2=sig2)
        g_preds.append(1 if g1 > g2 else 0)
        # print('g1', g1)
        # print('g2', g2)


    all_res.extend([{
        'name': 'Bayes Naive',
        'n_dims': d,
        'n_hidden': n_us,
        'acc_unseen': np.mean(np.array(g_preds) == ys)
    }])

df_bayes = pd.DataFrame(all_res)
df_bayes
# %%
adf = pd.concat((mdf, df_bayes))
gs = sns.relplot(adf, x='n_hidden', y='acc_unseen', col='n_dims', hue='name', col_wrap=4, kind='line', marker='o')

for g in gs.axes:
    g.set_xscale('log', base=2)

plt.savefig('fig/bayes_naive_rf.png')
    

# <codecell>
### PROPER SWEEP
def pred_acc(n_points, a_raw=1.5):
    if n_points == 2:
        return 0.75

    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = n_points * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)

sig2 = 1e-10

all_n_points = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# all_n_points = [2, 3]

all_res = []

for n_points in tqdm(all_n_points):
    n_dims = 64
    n_hidden = 1024

    gamma0 = 1
    gamma = gamma0 * np.sqrt(n_hidden)
    lr = gamma0 * 10

    n_patches = 2

    train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
    test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

    config = MlpConfig(mup_scale=True,
                    as_rf_model=False,
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
                        test_every=1000,
                        train_iters=10_000,
                        # lr=1e-3,
                        # optim=sign_sgd,
                        optim=optax.sgd,
                        lr=lr,
                        gamma=gamma,
                        seed=None)

    xs, ys = next(test_task)

    g_preds = []
    m_preds = []

    for x in xs:
        z1, z2 = x
        g1 = log_gen_like_same(z1, z2, n_dims, sig2)
        g2 = log_gen_like_diff(z1, z2, n_dims, sig2)
        g_preds.append(1 if g1 > g2 else 0)

        m1 = log_mem_like_same(z1, z2, n_dims, sig2, train_task.symbols)
        m2 = log_mem_like_diff(z1, z2, n_dims, sig2, train_task.symbols)
        m_preds.append(1 if m1 > m2 else 0)


    all_res.extend([{
        'name': 'Bayes Gen',
        'n_symbols': n_points,
        'n_dims': n_dims,
        'acc_unseen': np.mean(g_preds == ys)
    }, {
        'name': 'Bayes Mem',
        'n_symbols': n_points,
        'n_dims': n_dims,
        'acc_unseen': np.mean(m_preds == ys)
    }, {
        'name': 'Rich MLP',
        'n_symbols': n_points,
        'n_dims': n_dims,
        'acc_unseen': hist['test'][-1].accuracy.item()
    }, {
        'name': 'Theory',
        'n_symbols': n_points,
        'n_dims': n_dims,
        'acc_unseen': pred_acc(n_points)
    }])

# %%
plot_df = pd.DataFrame(all_res)
plot_df

# <codecell>
sns.lineplot(plot_df, x='n_symbols', y='acc_unseen', hue='name', marker='o')
plt.savefig('fig/bayes_gen_mem_theory_closeup.png')
