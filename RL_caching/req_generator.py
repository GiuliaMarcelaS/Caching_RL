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


def independent_workload(num_files, seed=None, sigma=1.0):
    """
    Peso/custo por arquivo, DESACOPLADO da popularidade -- representa o
    "tamanho" ou "custo de servir" de cada arquivo (ex.: bytes, tempo de CPU
    por invocacao), para simular sistemas onde um item popular nao e
    necessariamente barato de servir, e vice-versa.

    Distribuicao log-normal, normalizada para media 1.0 (assim, um cache com
    capacity=N continua representando "cerca de N itens de tamanho medio",
    facilitando comparar contra o caso uniforme onde todo arquivo pesa
    exatamente 1). sigma controla a variabilidade: sigma=0 equivale ao caso
    uniforme; sigma=1.0 ja da uma dispersao razoavel (itens podem variar por
    mais de uma ordem de grandeza entre si).

    Usa um stream de aleatoriedade PROPRIO, deslocado do seed base, para
    garantir que o resultado nao fica correlacionado com a permutacao de
    popularidades feita por zipf() mesmo reaproveitando o mesmo seed
    numerico -- ou seja, um arquivo popular tem a mesma chance de ser
    "leve" ou "pesado" que um arquivo raro.
    """
    if sigma <= 0:
        return np.ones(num_files)
    rng = np.random.default_rng(None if seed is None else seed + 1_000_003)
    w = rng.lognormal(mean=0.0, sigma=sigma, size=num_files)
    return w / w.mean()