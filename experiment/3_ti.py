"""Experiments with transitive inference"""

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
from model.transformer import TransformerConfig, SimpleTransformerConfig
from task.same_different import SameDifferent 
from task.ti import TiTask

n_symbols = 10

gamma = 10_000_000
base_lr = 1e-2
# base_lr = 1e-5
# base_lr = 1e-4
lr = gamma * base_lr

train_task = TiTask(n_symbols=n_symbols, sep_dists=[3], one_hot_encode=True, add_pos_emb=False)
test_task = TiTask(n_symbols=n_symbols, sep_dists=[5], one_hot_encode=True, add_pos_emb=False)

config = MlpConfig(mup_scale=False,
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=256, 
                   feature_learning_strength=gamma,
                   use_bias=False,
                   act_fn='relu')

# config = TransformerConfig(n_layers=1,
#                            n_hidden=256,
#                            pos_emb=True,
#                            n_mlp_layers=0,
#                            n_heads=1,
#                            layer_norm=False,
#                            as_rf_model=False,
#                            residual_connections=False,
#                            use_simple_att=True,
#                            freeze_emb=False,
#                            gamma=gamma)

# config = SimpleTransformerConfig(n_hidden=256, gamma=gamma)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=100_000, 
                    lr=lr,
                    optim=optax.sgd,
                    seed=None)

# %%
xs, ys = next(test_task)

preds = state.apply_fn({'params': state.params}, xs)
print((preds>0).astype(int) == ys)
print(preds)
print(ys)

# <codecell>
jax.tree.map(jnp.shape, state.params)

W = np.array(state.params['Dense_0']['kernel'])
a = np.array(state.params['DenseGeneral_0']['kernel'].squeeze())

a_norms = np.linalg.norm(a.squeeze(), axis=0)
a_sort_idx = np.argsort(a_norms)

# a[:,a_sort_idx[:255]] = 0
# a
# <codecell>
x = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]])
x = np.expand_dims(x, axis=0)

h = jax.nn.relu(x @ W)

att = np.einsum('bih,bjh->bij', h, h)
final_att = att[...,-1]  # batch x len
print(final_att)

h2 = np.einsum('bl,blh->blh', final_att, h)
out = np.sum(h2.squeeze() * a) / gamma
print(out)

# <codecell>
max_idxs = a_sort_idx[-50:]

# plt.imshow(W[:,max_idxs] * a[0,max_idxs])
plt.imshow(W[:,max_idxs])
plt.colorbar()

# plt.plot(np.sum(W[:,max_idxs], axis=-1), '--o')

# <codecell>
A = jax.nn.relu(W) @ jax.nn.relu(W.T)
# A = W @ W.T
plt.imshow(np.log(A))
plt.colorbar()

# <codecell>
plt.plot(a[0,a_sort_idx])
plt.plot(a[1,a_sort_idx])
# plt.axvline(x=245)


# <codecell>
state.apply_fn({'params': state.params}, x)

# <codecell>
### MLP
jax.tree.map(jnp.shape, state.params)

W = np.array(state.params['Dense_0']['kernel'])
a = np.array(state.params['Dense_1']['kernel'].squeeze())

a_sort_idx = np.argsort(a)

# a[:,a_sort_idx[:255]] = 0
# a

# <codecell>
# max_idxs = a_sort_idx[-30:]
max_idxs = a_sort_idx[:30]

# plt.imshow(W[:,max_idxs] * a[0,max_idxs])
plt.imshow(W[:,max_idxs])
plt.colorbar()

# plt.plot(np.sum(W[:,max_idxs], axis=-1), '--o')

# <codecell>
sel = W[:,a_sort_idx[0]]
plt.imshow(sel.reshape(2, -1).T)


# <codecell>
A = jax.nn.relu(W) @ jax.nn.relu(W.T)
# A = W @ W.T
plt.imshow(A)
plt.colorbar()

# <codecell>
plt.plot(a[a_sort_idx])
# plt.axvline(x=245)


# <codecell>
state.apply_fn({'params': state.params}, x)
