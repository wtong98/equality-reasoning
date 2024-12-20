"""Simulation of rich regime dynamics"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


# RICH REGIME MARKOV MODEL SIMULATION
def is_match(w1, w2, z1, z2):
    if w1[z1] + w2[z2] > 0:
        return True
    elif w1[z1] + w2[z2] == 0:
        if np.random.random() >= 0.5:
            return True
    
    return False

n_rep = 500
meas = []

for _ in tqdm(range(n_rep)):
    n_iters = 100
    n_vocab = 6
    batch_size = 100

    a = 1

    w1 = np.zeros(n_vocab)
    w2 = np.zeros(n_vocab)

    mu_count_0 = 0
    mu_total_0 = 0

    mu_count_pos = 0
    mu_total_pos = 0

    mu_count_neg = 0
    mu_total_neg = 0

    for i in range(n_iters):
        # if i < 1_000:
        #     mu_count_0 = 0
        #     mu_total_0 = 0

        #     mu_count_pos = 0
        #     mu_total_pos = 0

        #     mu_count_neg = 0
        #     mu_total_neg = 0

        bu1 = np.zeros(n_vocab)
        bu2 = np.zeros(n_vocab)

        for _ in range(batch_size):
            z1, z2 = np.random.choice(n_vocab, size=2, replace=False)
            if np.random.random() >= 0.5:
                z1 = z2
            elif z1 == z2:
                print('WHOOPS!')

            
            s1 = w1[z1] + w1[z2]
            s2 = w2[z1] + w2[z2]

            if s1 > 0 and s2 > 0:
                mu_total_pos += 1

            if s1 < 0 and s2 < 0:
                mu_total_neg += 1

            if s1 * s2 < 0:
                mu_total_0 += 1

            if is_match(w1, w2, z1, z2):
                if z1 == z2:
                    bu1[z1] += a
                    bu2[z2] += a
                else:
                    bu1[z1] -= a
                    bu2[z2] -= a

                if s1 > 0 and s2 > 0:
                    mu_count_pos += 1

                if s1 < 0 and s2 < 0:
                    mu_count_neg += 1

                if s1 * s2 < 0:
                    mu_count_0 += 1

        w1 += bu1
        w2 += bu2

        
        # print(f'ZS: {z1} {z2}')
        # print('W1', w1)
        # print('W2', w2)
        # print('---')

    # print('w1', np.sort(w1))
    # print('w2', np.sort(w2))
    # print('w1', w1)
    # print('w2', w2)

    # print('POS', mu_count_pos / mu_total_pos)
    # print('NEG', mu_count_neg / mu_total_neg)
    # print('ZER', mu_count_0 / mu_total_0)

    if mu_total_pos == 0 or mu_total_neg == 0 or mu_total_0 == 0:
        continue


    acc = 0

    for _ in range(batch_size):
        z1, z2 = np.random.choice(n_vocab, size=2, replace=False)
        if np.random.random() >= 0.5:
            z1 = z2
        
        acc += jax.nn.relu(w1[z1] + w2[z2])

    meas.append({
        'pos': mu_count_pos / mu_total_pos,
        'neg': mu_count_neg / mu_total_neg,
        'zer': mu_count_0 / mu_total_0,
        'magn': acc / batch_size,
        'abs': np.mean(np.abs(w1))
    })


# %%
df = pd.DataFrame(meas)
sns.histplot(df['pos'])
sns.histplot(df['neg'])
sns.histplot(df['zer'])

plt.axvline(x=5/10, color='red')

# <codecell>
df['magn'].mean()

# <codecell>
n_vocab = 100
w1 = [1] * 50 + [-1] * 50
# w2 = [1] * 50 + [-1] * 50
w2 = [-1] * 50 + [1] * 50

acc = 0
batch_size = 10000

pos = 0

for _ in range(batch_size):
    z1, z2 = np.random.choice(n_vocab, size=2, replace=False)
    if np.random.random() > 0.5:
        z1 = z2
    
    if z1 == z2:
        pos += 1
    
    acc += jax.nn.relu(w1[z1] + w2[z2])

pos / batch_size
acc / batch_size


# %%
### MEASURING VALUES IN REAL NETWORK
n_dims = 128
n_points = 100
n_hidden = 512

noise = 0

gamma0 = 10
gamma = gamma0 * np.sqrt(n_hidden)
lr = gamma0 * 10

n_patches = 2

train_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True, batch_size=1, noise=noise)
test_task = SameDifferent(n_patches=n_patches, n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024, noise=noise)

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
                    train_iters=5000,
                    # optim=sign_sgd,
                    # lr=1e-3,
                    optim=optax.sgd,
                    lr=lr,
                    gamma=gamma,
                    seed=None)

# %%
jax.tree.map(np.shape, state.params)

W = state.params['Dense_0']['kernel']
a = state.params['Dense_1']['kernel'].flatten()
sort_idxs = np.argsort(a)
a[sort_idxs[-1]]

W1, W2 = W[:n_dims], W[n_dims:]

zs = train_task.symbols

W1_proj = zs @ W1
W2_proj = zs @ W2

print(W1_proj[:,sort_idxs[-10]])
print(W2_proj[:,sort_idxs[-10]])

W1_pos = W1_proj[:,a>0]
np.sort(np.sum(W1_pos > 0, axis=0))

# <codecell>
z1, z2 = train_task.symbols[[1, 2]]
# z1 @ W1[:,sort_idxs[-1]]
# z2 @ W2[:,sort_idxs[-1]]

x = np.concatenate((z1, z2))[None]
jax.nn.relu(x @ W[:,sort_idxs[-1]]) * a[sort_idxs[-1]]

# state.apply_fn({'params': state.params}, x)

# jax.tree.map(jnp.shape, state.params)

# x @ state.params['Dense_0']['kernel'] @ state.params['Dense_1']['kernel']


# <codecell>
W1_proj[:,sort_idxs[-1]]
W2_proj[:,sort_idxs[0]]


# W1_pos[:,-1]

# <codecell>
np.mean(a[a < 0]) / np.mean(a[a > 0])
# a[sort_idxs[-128]]
# a[sort_idxs[128]]

# <codecell>
xs, _ = next(train_task)
xs = xs.reshape(xs.shape[0], -1)
actv = xs @ W

print(jax.nn.relu(actv[:,a<0]).mean())
print(jax.nn.relu(actv[:,a>0]).mean())

# <codecell>
train_task.batch_size = 1024
xs, ys = next(train_task)

xs_pos = xs[ys == 1]
out = state.apply_fn({'params': state.params}, xs_pos)
plt.hist(out)

xs_neg = xs[ys == 0]
out = state.apply_fn({'params': state.params}, xs_neg)
plt.hist(-out, alpha=0.5)



