"""
Match task accuracies
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.mlp import MlpConfig 
from task.same_different import SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000
n_vocab = np.round(2**np.linspace(4, 9, 20)).astype(int)
log10_gs = [-5, 0]
n_dims = np.round(2**np.linspace(4, 9, 20)).astype(int)
n_widths = [4096]
base_lr = 10

n_layers = 1
sig2 = 0

### START TEST CONFIGS
# run_split = 1

# train_iters = 2500
# n_vocab = [4]
# n_dims = [2]
# log10_gs = [0]
# n_widths = [256]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for n_hidden, d, v in itertools.product(n_widths, n_dims, n_vocab):
    noise = sig2
    for log10_gamma0 in log10_gs:
        gamma0 = 10**log10_gamma0
        gamma = gamma0 * np.sqrt(n_hidden)
        lr = gamma0**2 * base_lr

        all_cases.append(
            Case(rf'$\gamma=10^{ {log10_gamma0} }$',
                    MlpConfig(mup_scale=True, n_out=1, n_layers=1, n_hidden=n_hidden, use_bias=False),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce',
                                'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
                    train_task=SameDifferent(n_symbols=v, n_dims=d, noise=noise),
                    test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024, noise=noise),
                    info={'log10_gamma0': log10_gamma0})
        )

all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    case.train_task.symbols = None
    case.test_task.symbols = None
    # case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
