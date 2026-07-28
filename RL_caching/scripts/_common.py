import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import Optimal_QLRU
from monitor import Monitor
from q_learning import run_qlearning
from req_generator import independent_workload, zipf
from traffic_real import empirical_popularity_from_trace

QLEARN_MODES = ["qlearn", "lru", "oracle", "nocache"]
WORKLOAD_MODES = ["uniform", "independent"]


def build_popularities(traffic_type, catalog_size, alpha, real_trace_path, seed):
    if traffic_type == "irm":
        return zipf(alpha, catalog_size, seed=seed)
    elif traffic_type == "real":
        return empirical_popularity_from_trace(real_trace_path, catalog_size, seed=seed)
    raise ValueError(f"traffic_type desconhecido: {traffic_type!r}")


def build_workload(workload_mode, catalog_size, seed, workload_sigma=1.0):
    """
    Custo por arquivo usado para pesar a carga gerada por um cache miss.

    "uniform": todo arquivo pesa 1 (comportamento historico -- popularidade
        e usada como se fosse a propria carga, ja que cada requisicao conta
        igual).
    "independent": custo sorteado de uma log-normal, DESACOPLADO da
        popularidade (ver req_generator.independent_workload). Simula
        sistemas onde arquivos populares nao sao necessariamente baratos de
        servir, e vice-versa. Nota: hoje isso vale tanto para traffic_type
        "irm" quanto "real", ja que o pipeline de trafego real
        (traffic_real.py/preprocess_azure_trace.py) so extrai frequencia de
        invocacao do trace do Azure, nao duracao/memoria reais -- entao nao
        ha, por enquanto, uma nocao de "carga real" vinda do dataset em si.
    """
    if workload_mode == "uniform":
        return np.ones(catalog_size)
    elif workload_mode == "independent":
        return independent_workload(catalog_size, seed=seed, sigma=workload_sigma)
    raise ValueError(f"workload_mode desconhecido: {workload_mode!r} (use um de {WORKLOAD_MODES})")


def angle_from_jfi(jfi: float) -> float:
    jfi_clamped = min(max(jfi, 0.0), 1.0)
    return float(np.degrees(np.arccos(np.sqrt(jfi_clamped))))


def run_one_config(traffic_type, num_servers, catalog_size, num_requests, capacity,
                    run_id, alpha, beta, real_trace_path, skew_fraction=0.0,
                    workload_mode="uniform", workload_sigma=1.0):
    """Roda Optimal_QLRU + as 4 variantes de qlearning_balancer

    skew_fraction: repassado para Monitor -- fracao do catalogo sempre fixa
        no servidor 0. Default 0.0 = distribuicao totalmente aleatoria entre
        os servidores (sem hotspot estrutural). Veja o docstring de
        Monitor.__init__ em monitor.py para mais detalhes.

    workload_mode / workload_sigma: ver build_workload() acima. Controla
        apenas quanta carga um MISS gera no servidor (usado no calculo do
        JFI). Nao afeta a capacidade do cache -- o parametro `sizes` do
        Optimal_QLRU continua uniforme, entao "capacity" sempre significa
        "cerca de N itens", nos dois modos.
    """
    seed = run_id
    rng_py = random.Random(seed)

    popularities = build_popularities(traffic_type, catalog_size, alpha, real_trace_path, seed)
    req = np.random.default_rng(seed).choice(catalog_size, num_requests, p=popularities)
    workload = build_workload(workload_mode, catalog_size, seed, workload_sigma)

    monitor = Monitor(num_servers, catalog_size, seed=seed, skew_fraction=skew_fraction)

    rows = []

    # --- Optimal_QLRU (cache.py) ---
    # sizes fica uniforme de proposito: workload aqui so pesa a carga gerada
    # no servidor (via monitor.run_metrics), nao a ocupacao do cache.
    initial_state = rng_py.sample(range(catalog_size), min(capacity, catalog_size))
    sizes = [1] * catalog_size
    opt_cache = Optimal_QLRU(initial_state, capacity, beta, sizes, monitor.file_to_server)
    res_opt = monitor.run_metrics(opt_cache, req, popularities, capacity, algorithm="Optimal_QLRU",
                                   plot=False, workload=workload)
    rows.append({
        "algorithm": "Optimal_QLRU",
        "hit_rate": res_opt["hit_rate"],
        "jfi": res_opt["jfi"],
        "angle_degrees": res_opt["angle_degrees"],
    })

    # --- Q-learning + baselines ---
    for mode in QLEARN_MODES:
        res_q = run_qlearning(monitor, req, catalog_size, capacity, mode=mode, seed=seed, workload=workload)
        rows.append({
            "algorithm": f"qlearning_balancer[{mode}]",
            "hit_rate": res_q["hit_rate"],
            "jfi": res_q["jfi"],
            "angle_degrees": angle_from_jfi(res_q["jfi"]),
        })

    common = {
        "traffic_type": traffic_type,
        "num_servers": num_servers,
        "capacity": capacity,
        "num_requests": num_requests,
        "run_id": run_id,
        "skew_fraction": skew_fraction,
        "workload_mode": workload_mode,
    }
    for row in rows:
        row.update(common)
    return rows