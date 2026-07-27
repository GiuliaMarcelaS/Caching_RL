import numpy as np

def Uniform(F, seed=None):
    p = np.ones(F)/F
    if seed is not None:
        pass
    return p

def zipf(alpha, num_files, seed=None):
    p = np.array([(i+1)**-alpha for i in range(num_files)])
    p = p/sum(p)
    if seed is not None:
        rng = np.random.default_rng(seed)
        p = rng.permutation(p)
    return p
