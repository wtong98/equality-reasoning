"""Same-different tasks"""

# <codecell>

import numpy as np

from pentomino import pieces
pieces = np.array(pieces, dtype='object')

def batch_choice(a, n_elem, batch_size, rng=None):
    if type(a) == int:
        a = np.arange(a)

    assert n_elem <= len(a), f'require n_elem <= len(a), got n_elem={n_elem} and len(a)={len(a)}'

    if rng is None:
        rng = np.random.default_rng(None)

    idxs = np.tile(a, (batch_size, 1))
    idxs = rng.permuted(idxs, axis=1)
    idxs = idxs[:,:n_elem]
    return idxs


class SameDifferentPentomino:
    def __init__(self, width=2, batch_size=128):
        self.width = width
        self.batch_size = batch_size
        self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = np.zeros((self.batch_size, self.width * 5, self.width * 5))
        xs_idxs = batch_choice(len(pieces), 2, self.batch_size)
        xs_patches = batch_choice(self.width**2, 2, self.batch_size)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs_idxs[idxs,1] = xs_idxs[idxs,0]
        
        xs_pieces = pieces[xs_idxs]
        for x, x_patch, x_piece in zip(xs, xs_patches, xs_pieces):
            a, b = x_piece
            a = self.rng.choice(a)
            b = self.rng.choice(b)

            a_x = 5 * (x_patch[0] // self.width)
            a_y = 5 * (x_patch[0] % self.width)

            b_x = 5 * (x_patch[1] // self.width)
            b_y = 5 * (x_patch[1] % self.width)

            x[a_x:a_x+5, a_y:a_y+5] = a
            x[b_x:b_x+5, b_y:b_y+5] = b

        return xs, ys


    def __iter__(self):
        return self

# rng = np.random.default_rng(None)
# rng.choice(np.array(pieces, dtype='object')[[[1,2], [2,3], [3,4]]][0][0])
# task = SameDifferentPentomino(batch_size=1)
# xs, ys = next(task)

# xs = xs.squeeze()

# import matplotlib.pyplot as plt
# plt.imshow(xs)
# print(ys)


# <codecell>

class SameDifferent:
    def __init__(self, n_symbols=None, task='hard',
                 n_dims=2, thresh=0, radius=1,    # soft/hard params
                 n_seen=None, sample_seen=True,   # token params
                 seed=None, reset_rng_for_data=True, batch_size=128) -> None:

        if task == 'token':
            assert n_symbols is not None and n_symbols >= 4, 'if task=token, n_symbols should be >= 4'
            
            if n_seen is None:
                n_seen = n_symbols // 2

        self.n_symbols = n_symbols
        self.task = task
        self.n_dims = n_dims
        self.thresh = thresh
        self.radius = radius
        self.n_seen = n_seen
        self.sample_seen = sample_seen
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if self.n_symbols is not None:
            self.symbols = self.rng.standard_normal((self.n_symbols, self.n_dims)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.task == 'soft':
            return self._sample_soft()
        elif self.task == 'hard':
            return self._sample_hard()
        elif self.task == 'token':
            return self._sample_token()
        else:
            raise ValueError(f'unrecognized task type: {self.task}')

    def _sample_soft(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        # norms = np.linalg.norm(xs, axis=-1, keepdims=True)
        # xs = xs / norms * self.radius

        x0, x1 = xs[:,0], xs[:,1]
        ys = (np.einsum('bi,bi->b', x0, x1) > self.thresh).astype('float')
        return xs, ys.flatten()
    
    def _sample_hard(self):
        if self.n_symbols is None:
            xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        else:
            sym_idxs = batch_choice(np.arange(self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)
            xs = self.symbols[sym_idxs]

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys
    
    def _sample_token(self):
        if self.sample_seen:
            xs = batch_choice(np.arange(0, self.n_seen), 2, batch_size=self.batch_size, rng=self.rng)
        else:
            xs = batch_choice(np.arange(self.n_seen, self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self