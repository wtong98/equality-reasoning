"""Prepare cifar100 task"""

# <codecell>
import numpy as np
from pathlib import Path
import pickle

import sys
sys.path.append('../')
from common import *
from model.cnn import CnnConfig

# <codecell>
def load_data(preprocess_cnn=True, seed=None):
    if seed is None:
        seed = new_seed()

    parent_path = Path(__file__).parent / 'dataset'
    dataset_path = parent_path / 'cifar100'

    with open(dataset_path / 'train', 'rb') as fp:
        train_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'test', 'rb') as fp:
        test_set = pickle.load(fp, encoding='bytes')

    with open(dataset_path / 'meta', 'rb') as fp:
        meta_set = pickle.load(fp)


    train_labs = train_set[b'fine_labels']
    train_data = train_set[b'data']

    test_labs = test_set[b'fine_labels']
    test_data = test_set[b'data']

    all_labs = train_labs + test_labs
    all_data = np.concatenate((train_data, test_data), axis=0)
    all_data = (all_data - all_data.mean()) / all_data.std()

    if preprocess_cnn:
        all_data = all_data.reshape(-1, 32, 32, 3, order='F')
        model = CnnConfig(cnn_widths=[32, 64, 128], headless=True).to_model()
        # params = model.init(jax.random.PRNGKey(seed), all_data[[0]])['params']

        with open(parent_path / 'cnn_params.pkl', 'rb') as fp:
            params = pickle.load(fp)

        batches = []
        for b in np.split(all_data, 50):
            out = model.apply({'params': params}, b)
            batches.append(out)

        all_data = np.concatenate(batches, axis=0)
        all_data = all_data.reshape(all_data.shape[0], -1)

    label_names = meta_set['fine_label_names']

    return {
        'data': all_data,
        'labels': all_labs,
        'names': label_names
    }

# res = load_data()


# # <codecell>
# res['data'].shape

# # <codecell>
# import matplotlib.pyplot as plt

# all_data = res['data']

# ims = all_data.reshape(-1, 32, 32, 3, order='F')

# plt.imshow(ims[6])


# # %%

# a = np.zeros((100, 32, 32))
# np.split(a, 5)[0].shape
