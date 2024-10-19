"""
Big patch number
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
n_vocab = 2**np.arange(3, 15)
log10_gs = []
n_dims = [2, 4, 8, 16, 32, 64, 128, 256]
ks = [2]
base_lr = 1

n_layers = 1
n_widths = 2**np.arange(7, 17)

### START TEST CONFIGS
# run_split = 1

# train_iters = 2500
# n_vocab = [4]
# n_dims = [2]
# log10_gs = [0]
# ks = [2]
# n_widths = [128]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for v, d, k, n_hidden in itertools.product(n_vocab, n_dims, ks, n_widths):
    params = {'n_symbols': v, 'n_dims': d}
    
    all_cases.extend([
        Case(f'RF', 
            MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden, as_rf_model=True),
            train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
            train_task=SameDifferent(n_patches=k, n_symbols=v, n_dims=d),
            test_task=SameDifferent(n_patches=k, n_symbols=None, n_dims=d, batch_size=1024)),
    ])

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
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
