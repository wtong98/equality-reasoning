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

run_split = 3

train_iters = 20_000
n_vocab = np.round(2**np.linspace(2, 10, 21)).astype(int)
n_dims = [2, 4, 8, 16, 32, 64, 128]
gs = [0.01, 0.1, 1]
base_lr = 1

n_layers = 1
n_hidden = 512

### START TEST CONFIGS
# run_split = 1

# train_iters = 2500
# n_vocab = [4]
# n_dims = [2]
# gs = [1]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for v in n_vocab:
    for d in n_dims:
        params = {'n_symbols': v, 'n_dims': d}
        
        all_cases.extend([
            Case(f'MLP (RF)', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden, as_rf_model=True),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),

            Case(f'MLP (Adam)', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden),
                train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),
        ])

        for g in gs:
            gamma = g * np.sqrt(n_hidden)
            lr = g**2 * base_lr

            all_cases.append(
                Case(rf'MLP ($\gamma_0={g}$)',
                     MlpConfig(mup_scale=True, n_out=1, n_layers=1, n_hidden=n_hidden),
                     train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce',
                                 'optim': optax.sgd, 'lr': lr, 'gamma': gamma},
                     train_task=SameDifferent(n_symbols=v, n_dims=d),
                     test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024),
                     info={'gamma0': g})
            )

all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
