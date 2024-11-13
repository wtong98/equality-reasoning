"""More careful dissection of rich MLP"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm, halfnorm
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig
from task.same_different import SameDifferent 

# <codecell>
n_dims = 128
n_points = 4
n_hidden = 4096

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

n_patches = 2

train_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=128)
test_task = SameDifferent(noise=0, n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

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
                    train_iters=5_000,
                    # lr=1e-3,
                    # optim=sign_sgd,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# <codecell>
xs, ys = next(test_task)
out = state.apply_fn({'params': state.params}, xs)
preds = (out > 0).astype(bool)

print(np.mean(ys == preds))
print('---')
print(np.mean(ys[ys>0] == preds[ys>0]))
print(np.mean(ys[ys==0] == preds[ys==0]))

print('---')
print(np.mean(ys[preds>0] == preds[preds>0]))
print(np.mean(ys[preds==0] == preds[preds==0]))

print('---')
print(np.mean(ys))
print(np.mean(preds))

# <codecell>
### Rich with noise
def pred_pos_acc(L, sig2, neg_dim_guess_prefac=0):
    a = 1.4**2  # NOTE: a seems to be an effectively tunable free parameter (not good)

    neg_dim_guess = neg_dim_guess_prefac * (1 + 2*sig2)
    prefactor = np.sqrt(2 / (np.pi - 2))

    t1 = np.sqrt(sig2 + 1) - a * np.sqrt(sig2 + neg_dim_guess)
    t2 = np.sqrt((a**2 + 1) * sig2 + 1 + a**2 * neg_dim_guess)
    t = t1 / t2

    z = prefactor * t * np.sqrt(L)
    return norm.cdf(z)

L = 10
p = 0
sig2 = 0.1
pred_pos_acc(L, sig2, neg_dim_guess_prefac=p)

# <codecell>
def pred_neg_acc(L, sig2=0, l_adjust=1):  # TODO: incorporate sig2
    a_raw = 1.5**2
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = np.sqrt(L - l_adjust) * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)

pred_neg_acc(5, l_adjust=1)
# <codecell>
W = np.array(state.params['Dense_0']['kernel'])
a = np.array(state.params['Dense_1']['kernel'])

n_iters = 10_000
xs = np.random.randn(n_iters, 2*n_dims) / np.sqrt(n_dims)
xs[:,:n_dims] = xs[:,n_dims:]

xs += np.random.randn(*xs.shape) * np.sqrt(sig2/n_dims)

out = jax.nn.relu(xs @ W) @ a
np.mean(out > 0)

# <codecell>
af = a.flatten()
sidx = np.argsort(af)

res = xs @ W
res_p = res[:,sidx[-1]]
res_n = res[:,sidx[0]]

plt.hist(res_p, bins=30, alpha=0.4, density=True)
plt.hist(res_n, bins=30, alpha=0.4, density=True)

xlins = np.linspace(-2, 2, 1000)

m = np.mean(res, axis=0)
s = np.std(res, axis=0)

# ys = norm.pdf(xlins, loc=0, scale=np.sqrt(0.21 * af[sidx[-1]] * 2 * L * (sig2/n_dims + 1/n_dims)))
# zs = norm.pdf(xlins, loc=0, scale=np.sqrt(0.21 * -af[sidx[0]] * 2 * L * (sig2/n_dims)))
ys = norm.pdf(xlins, loc=0, scale=np.sqrt(1.1 * 2 * L * (sig2/n_dims + 1/n_dims)))
zs = norm.pdf(xlins, loc=0, scale=np.sqrt(1.1 * 1.5 * 2 * L * (sig2/n_dims)))
# ys = norm.pdf(xlins, loc=0, scale=s[sidx[-1]])
# zs = norm.pdf(xlins, loc=0, scale=s[sidx[0]])

plt.plot(xlins, ys)
plt.plot(xlins, zs)

# af[sidx[0]] / af[sidx[-1]]
np.mean(af[af < 0]) / np.mean(af[af > 0])

# <codecell>
s_p = s[af > 0].mean()
s_n = s[af < 0].mean()

res = jax.nn.relu(xs @ W)
res_p = res[:, af > 0].mean(axis=1)
res_n = res[:, af < 0].mean(axis=1)

plt.hist(res_p, bins=30, alpha=0.4, density=True)
plt.hist(res_n, bins=30, alpha=0.4, density=True)

xlins = np.linspace(0, 0.1, 1000)

ps = norm.rvs(size=(n_iters, L), loc=0, scale=s_p)
ps = (np.abs(ps) / 2).mean(axis=1)
plt.hist(ps, alpha=0.3, bins=30, density=True)

# ps = norm.pdf(xlins, 
#               loc=np.sqrt(2 / np.pi) * np.sqrt(2 * L) * np.sqrt((sig2 + 1) / n_dims), 
#               scale=(1 - 2 * np.pi) * 2 * (sig2 + 1) / n_dims)

# ps = norm.pdf(xlins, loc=s_p / np.sqrt(L), scale=s_p / np.sqrt(L))


# ys = halfnorm.pdf(xlins, loc=0, scale=np.sqrt(0.17 * af[sidx[-1]] * 2 * L * (sig2/n_dims + 1/n_dims)))
# zs = halfnorm.pdf(xlins, loc=0, scale=np.sqrt(0.17 * -af[sidx[0]] * 2 * L * (sig2/n_dims)))

# plt.plot(xlins, ps)
# plt.plot(xlins, zs)

np.mean(res_p - 1.5 * res_n > 0)

# <codecell>
out = res_p - 1.5 * res_n
out = out / np.std(out)
plt.hist(out, bins=30, density=True, alpha=0.3)

np.mean(out)

# t1 = np.sqrt(sig2 + 1) - 1.5 * np.sqrt(sig2)
# t2 = np.sqrt((1.5**2 + 1) * sig2 + 1)
# t = t1 / t2

# z = prefactor * t * np.sqrt(L)
# z


# <codecell>
# def pred_acc(n_points, a_raw=1.5):
#     a_raw = a_raw**2
#     a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
#     pt = np.sqrt(n_points) * np.sqrt(2 / (np.pi - 2)) * a
#     return norm.cdf(pt)

# pred_acc(8)

# %%
### Rich no noise case
W = np.array(state.params['Dense_0']['kernel'])
a = np.array(state.params['Dense_1']['kernel'])

n_iters = 10_000
xs = np.random.randn(n_iters, 2*n_dims) / np.sqrt(n_dims)

out = jax.nn.relu(xs @ W) @ a
np.mean(out < 0)

# <codecell>
res = (xs @ W)
idx = 10
plt.hist(res[:,idx], bins=20, density=True)

m = np.mean(res, axis=0)
s = np.std(res, axis=0)

res_sim = norm.rvs(loc=m[idx], scale=s[idx], size=n_iters)
plt.hist(res_sim, density=True, alpha=0.5, bins=20)

# <codecell>
W_p = W[:,a.flatten() > 0]
W_n = W[:,a.flatten() < 0]

res_pre = (xs @ W_n)[:,:].mean(axis=1) / 2
res = jax.nn.relu(xs @ W_n)[:,:].mean(axis=1)

plt.hist(res, bins=20, density=True)
plt.hist(res_pre, bins=20, density=True)
print('mean_pre', np.mean(res_pre))
print('std_pre', np.std(res_pre))
print('mean', np.mean(res))
print('std', np.std(res))

res_pre_sim = norm.rvs(loc=np.mean(res_pre), scale=np.std(res_pre), size=n_iters)
plt.hist(res_pre_sim, alpha=0.5, density=True, bins=20)

dot = xs @ W_n
m = np.mean(dot, axis=0)
s = np.std(dot, axis=0)


res_sim = np.zeros(n_iters)
# for i, (loc, scale) in enumerate(zip(m, s)):
#     samp = norm.rvs(loc=loc, scale=scale, size=n_iters) / W_p.shape[1]
#     # samp = dot[:,i] / W_p.shape[1]
#     res_sim += np.abs(samp) / 2

# k = n_points - 1
k = 4
for i in range(k):
    samp = norm.rvs(loc=np.mean(m), scale=np.mean(s), size=n_iters) / k
    # samp = dot[:,i] / W_p.shape[1]
    res_sim += np.abs(samp) / 2
    # res_sim += samp

# res_sim = res_pre_sim[res_pre_sim > 0]
# res_sim = np.abs(res_pre_sim)
g = plt.hist(res_sim, density=True, bins=20)

print('mean_sim', np.mean(res_sim))
print('std_sim', np.std(res_sim))

# <codecell>
z1s = W_n[:n_dims, :]
overlap = train_task.symbols @ z1s
plt.imshow(overlap[:,:50].T, cmap='plasma', vmin=-1, vmax=1)
plt.colorbar()

# <codecell>
out = np.std(res) / np.std(res_sim)
out

# <codecell>
L = 5
zs = np.random.randn(n_iters, n_dims) * np.sqrt(2/n_dims)

symbols = np.random.randn(L, n_dims) / np.sqrt(n_dims)
symbols.shape

overlap = zs @ symbols.T
out = np.sum(overlap, axis=1)
out = np.abs(out) / 2

plt.hist(out)


# <codecell>
idx = 19
loc, scale = m[idx], s[idx]

samp = norm.rvs(loc=loc, scale=scale, size=n_iters)
true = dot[:,idx]

plt.hist(true, bins=20, density=True)
plt.hist(samp, bins=20, density=True, alpha=0.5)

# <codecell>
res = jax.nn.relu(xs @ W_p).mean(axis=1) - 1.5 * jax.nn.relu(xs @ W_n).mean(axis=1)
np.mean(res < 0)

# <codecell>
### VALIDATE WITH EXPERIMENTS
df = collate_dfs('remote/9_rich_dissection/noise', concat=True)
df

# <codecell>
def extract_plot_vals(row):
    hist_acc = [m['accuracy'].item() for m in row['hist']['test']]
    hist_loss = [m['loss'].item() for m in row['hist']['test']]

    return pd.Series([
        row['name'],
        row['info']['log10_gamma0'] if 'log10_gamma0' in row['info'] else -10,
        row['info']['sig2'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        max(hist_acc),
        min(hist_loss),
    ], index=['name', 'gamma0', 'sig2', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen', 'acc_best', 'loss_best'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
def pred_pos_acc(L, sig2, neg_dim_guess_prefac=0):
    a = 1.4**2  # NOTE: a seems to be an effectively tunable free parameter (not good)

    neg_dim_guess = neg_dim_guess_prefac * (1 + 2*sig2)
    prefactor = np.sqrt(2 / (np.pi - 2))

    t1 = np.sqrt(sig2 + 1) - a * np.sqrt(sig2 + neg_dim_guess)
    t2 = np.sqrt((a**2 + 1) * sig2 + 1 + a**2 * neg_dim_guess)
    t = t1 / t2

    z = prefactor * t * np.sqrt(L)
    return norm.cdf(z)

def pred_neg_acc(L, sig2=0, l_adjust=1):  # TODO: incorporate sig2
    a_raw = 1.5**2
    a = (a_raw - 1) / (np.sqrt(a_raw**2 + 1))
    pt = np.sqrt(L - l_adjust) * np.sqrt(2 / (np.pi - 2)) * a
    return norm.cdf(pt)

def pred_acc(L, sig2):
    pos_acc = pred_pos_acc(L, sig2)
    neg_acc = pred_neg_acc(L, sig2)

    return (pos_acc + neg_acc) / 2


Ls = np.unique(plot_df['n_symbols'])
sig2s = np.unique(plot_df['sig2'])

res = []
for L, sig2 in itertools.product(Ls, sig2s):
    res.append({
        'n_symbols': L,
        'sig2': sig2,
        'acc_best': pred_acc(L, sig2)
    })

res_df = pd.DataFrame(res)
res_df

# <codecell>
g = sns.lineplot(plot_df, x='n_symbols', y='acc_best', hue='sig2', marker='o')
sns.lineplot(res_df, x='n_symbols', y='acc_best', hue='sig2')

g.set_xscale('log', base=2)
