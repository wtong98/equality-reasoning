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


n_hidden = 512

ps = np.random.permutation(np.arange(18))
n_train = 2

ps_train = ps[:n_train]
ps_test = ps[n_train:]

train_task = SameDifferentPentomino(ps=ps_train, width=2, batch_size=128)
test_task = SameDifferentPentomino(ps=ps_test, width=2, batch_size=128)

gamma0 = 10
lr = gamma0 * 0.1

config = MlpConfig(n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
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
                    train_iters=100_000, 
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
- TODO: try more extreme train-test splits
'''
