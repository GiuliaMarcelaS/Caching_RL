"""
Generalizaçao do qlearning para N servidores, com balanceamento de carga (Jain's Fairness Index) como recompensa.
"""
import random
from collections import OrderedDict

import numpy as np


def _freq_bin(cnt: int, thr: tuple) -> int:
    return 0 if cnt < thr[0] else (1 if cnt < thr[1] else 2)


def _state_index(balance: int, on_heaviest: int, cached: int, fb: int, B: int) -> int:
    return ((balance * 2 + on_heaviest) * 2 + cached) * B + fb


def run_qlearning(monitor, req, catalog_size: int, capacity: int, mode: str = "qlearn",
                   workload=None, B: int = 3, freq_thr=None,
                   alpha_q: float = 0.1, gamma: float = 0.9,
                   eps_start: float = 0.3, eps_end: float = 0.02,
                   balance_thr: float = 0.9, eta_A: float = 0.02,
                   seed: int = 0, curve_points: int = 200) -> dict:
    
    rng_py = random.Random(seed)
    T = len(req)
    if freq_thr is None:
        freq_thr = (max(2, int(0.3 * (T / catalog_size))), max(4, int(2.0 * (T / catalog_size))))

    num_servers = monitor.num_servers
    NS = 8 * B
    Q = np.zeros((NS, 2))
    A = np.zeros(num_servers)  # carga por servidor 
    freq = np.zeros(catalog_size, dtype=np.int64)
    cache: "OrderedDict[int, bool]" = OrderedDict()
    hits = 0
    jain_curve = []
    s_prev = a_prev = None
    curve_every = max(1, T // curve_points)

    for t in range(T):
        f = int(req[t])
        srv = monitor.get_server(f)
        freq[f] += 1
        cached = f in cache
        hit = cached

        heaviest = int(np.argmax(A))
        on_heaviest = 1 if srv == heaviest else 0
        balance = 1 if monitor.jains_fairness_index(A) >= balance_thr else 0
        s = _state_index(balance, on_heaviest, int(cached), _freq_bin(int(freq[f]), freq_thr), B)

        if mode == "qlearn":
            eps = max(eps_end, eps_start * (1 - t / T))
            a = rng_py.randint(0, 1) if rng_py.random() < eps else int(np.argmax(Q[s]))
        elif mode == "lru":
            a = 1 if not hit else 0
        elif mode == "nocache":
            a = 0
        elif mode == "oracle":
            a = (1 if srv == heaviest else 0) if not hit else 0
        else:
            raise ValueError(f"mode desconhecido: {mode!r}")

        inc = np.zeros(num_servers)
        if not hit:
            inc[srv] = 1.0 if workload is None else float(workload[f])
            if a == 1:
                if len(cache) >= capacity:
                    cache.popitem(last=False)  
                cache[f] = True
        else:
            hits += 1
            if a == 1:
                del cache[f]
            else:
                cache.move_to_end(f)

        A = (1 - eta_A) * A + eta_A * inc

        if mode == "qlearn" and s_prev is not None:
            r = monitor.jains_fairness_index(A)
            Q[s_prev, a_prev] += alpha_q * (r + gamma * np.max(Q[s]) - Q[s_prev, a_prev])
        s_prev, a_prev = s, a

        if t % curve_every == 0:
            jain_curve.append(monitor.jains_fairness_index(A))

    final_jain = monitor.jains_fairness_index(A)
    return {
        "algorithm": f"qlearning_balancer[{mode}]",
        "hit_rate": hits / T if T else 0.0,
        "jfi": final_jain,
        "omega": A.tolist(),
        "curve": jain_curve,
        "Q": Q,
    }
