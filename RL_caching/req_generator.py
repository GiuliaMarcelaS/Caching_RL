import numpy as np

def Uniform(F):
    p = np.ones(F)/F
    return p

def zipf(alpha, num_files):
    p = np.array([(i+1)**-alpha for i in range(num_files)])
    return p/sum(p)