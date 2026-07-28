"""
Generalizaçao do qlearning para N servidores, com balanceamento de carga (Jain's Fairness Index) como recompensa.

Nota de design: a recompensa usada no update de Q e *apenas* o Jain's Fairness
Index apos a acao (nao inclui hit rate). Isso e intencional: o objetivo deste
agente e aprender a balancear carga; o hit rate e reportado como metrica de
avaliacao a parte, nao como parte do reward. Se no futuro quiser um agente que
otimize os dois objetivos simultaneamente, e preciso somar um termo de hit
rate ao reward abaixo (e provavelmente reponderar com um beta, como no
Optimal_QLRU de cache.py).
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
    curve_every = max(1, T // curve_points)

    # s_prev/a_prev: estado e acao da transicao ainda nao usada no update de Q.
    # r_prev: recompensa observada logo apos a_prev ter sido aplicada (ou seja,
    # jfi(A) calculado no FIM da iteracao anterior, com a carga ja atualizada
    # por a_prev). E crucial que r_prev corresponda a a_prev, e nao a acao
    # escolhida na iteracao atual -- por isso ela e calculada e guardada ao
    # final de cada iteracao, e so consumida no comeco da iteracao seguinte.
    s_prev = a_prev = None
    r_prev = None

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

        # Update de Q para a transicao (s_prev, a_prev) -> s, usando a
        # recompensa r_prev observada exatamente apos a_prev ter sido
        # aplicada (e nao a recompensa da acao que sera escolhida agora).
        if mode == "qlearn" and s_prev is not None:
            Q[s_prev, a_prev] += alpha_q * (r_prev + gamma * np.max(Q[s]) - Q[s_prev, a_prev])

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

        # Recompensa gerada por "a" nesta iteracao -- so sera usada no update
        # de Q da PROXIMA iteracao, quando "a" ja tiver virado a_prev.
        if mode == "qlearn":
            r_prev = monitor.jains_fairness_index(A)
        s_prev, a_prev = s, a

        if t % curve_every == 0:
            jain_curve.append(monitor.jains_fairness_index(A))

    # Ultima transicao observada (s_prev, a_prev, r_prev) ainda nao entrou no
    # update de Q dentro do loop, pois isso so acontece no comeco da iteracao
    # SEGUINTE. Sem mais requisicoes, tratamos como transicao terminal (sem
    # bootstrap do proximo estado), para nao descartar essa amostra.
    if mode == "qlearn" and s_prev is not None:
        Q[s_prev, a_prev] += alpha_q * (r_prev - Q[s_prev, a_prev])

    final_jain = monitor.jains_fairness_index(A)
    return {
        "algorithm": f"qlearning_balancer[{mode}]",
        "hit_rate": hits / T if T else 0.0,
        "jfi": final_jain,
        "omega": A.tolist(),
        "curve": jain_curve,
        "Q": Q,
    }