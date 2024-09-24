"""
A simple transitive inference task

Paper implementation: https://github.com/sflippl/relational-generalization-in-ti/tree/main

author: William Tong (wtong@g.harvard.edu)
"""

'''
NOTE: Lippl's implementation concats one-hots directly
- may be solved by using a *learnable* embedding layer
'''

# <codecell>
import itertools
import numpy as np

class TiTask:
    """Adhere's closely to Lippl's implementation"""
    def __init__(self, n_symbols, sep_dists=None, one_hot_encode=True) -> None:
        self.n_symbols = n_symbols
        self.sep_dists = sep_dists

        self.xs = itertools.product(
                            range(n_symbols), 
                            range(n_symbols))

        self.xs = [(a, b) for a, b in self.xs if a != b]
        if sep_dists is not None:
            self.xs = [(a, b) for a, b in self.xs if np.abs(a - b) in sep_dists]
        self.ys = [1 if a > b else -1 for a, b in self.xs]

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)

        if one_hot_encode:
            # slick code golf from 
            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            self.xs = (np.arange(self.xs.max() + 1) == self.xs[...,None]).astype(float)
    
    def __next__(self):
        return self.xs, self.ys
    
    def __iter__(self):
        return self

task = TiTask(3, sep_dists=[1], one_hot_encode=True)
xs, ys = next(task)  # TODO: run and see if there is a difference in performance <-- STOPPED HERE

# def enumerate_pairs(n_items, dist):
#     if dist > n_items - 1:
#         raise ValueError(f'dist={dist} is too large for n_items={n_items}')
    
#     idxs = np.arange(n_items)
#     partner = idxs + dist
#     valid_idxs = partner < n_items

#     return np.stack((idxs[valid_idxs], partner[valid_idxs])).astype(np.int32)


# class TiTask:
#     def __init__(self, n_items=5, n_dims=100, dist=1, dist_p=None, batch_size=32) -> None:
#         self.rep = np.random.randn(n_items, n_dims)

#         try:
#             self.dist = list(dist)
#         except TypeError:
#             self.dist = [dist]
        
#         if dist_p is None:
#             dist_p = np.ones(len(self.dist))
#         self.dist_p = dist_p / np.sum(dist_p)

#         self.batch_size = batch_size
#         self.all_pairs = {d: enumerate_pairs(n_items, d) for d in self.dist}
    
#     def __next__(self):
#         dists = np.random.choice(self.dist, p=self.dist_p, size=self.batch_size)
#         pairs = [self.all_pairs[d] for d in dists]

#         def pick(pair):
#             idx = np.random.choice(pair.shape[1])
#             return pair[:, idx]

#         pairs = np.stack([pick(p) for p in pairs])
#         labs = np.random.choice((0, 1), size=self.batch_size)
#         rev_idx = labs == 0
#         pairs[rev_idx] = np.flip(pairs[rev_idx], axis=1)
#         return pairs, labs

#     def __iter__(self):
#         return self

# def get_test_data(n):
#     """Generate test data for relational tasks.
#     """
#     test_x = np.zeros((n, n, n, 2))
#     for i, j in itertools.product(range(n), range(n)):
#         test_x[i,j,i,0] = 1
#         test_x[i,j,j,1] = 1
#     return test_x

# def get_transitive_data(n):
#     """Generate training data for TI task.
#     """
#     test_x = get_test_data(n)
#     x = test_x[tuple(zip(*([(i, i+1) for i in range(n-1)] + [(i+1, i) for i in range(n-1)])))]
#     y = np.array([1.]*(n-1)+[-1.]*(n-1))
#     return x, y

# %%
