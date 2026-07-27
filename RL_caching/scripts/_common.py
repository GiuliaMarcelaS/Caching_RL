import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import Optimal_QLRU
from monitor import Monitor
from q_learning import run_qlearning
from req_generator import zipf
from traffic_real import empirical_popularity_from_trace

QLEARN_MODES = ["qlearn", "lru", "oracle", "nocache"]


def build_popularities(traffic_type, catalog_size, alpha, real_trace_path, seed):
    if traffic_type == "irm":
        return zipf(alpha, catalog_size, seed=seed)
    elif traffic_type == "real":
        return empirical_popularity_from_trace(real_trace_path, catalog_size, seed=seed)
    raise ValueError(f"traffic_type desconhecido: {traffic_type!r}")


def angle_from_jfi(jfi: float) -> float:
    jfi_clamped = min(max(jfi, 0.0), 1.0)
    return float(np.degrees(np.arccos(np.sqrt(jfi_clamped))))


def run_one_config(traffic_type, num_servers, catalog_size, num_requests, capacity,
                    run_id, alpha, beta, real_trace_path):
    """Roda Optimal_QLRU + as 4 variantes de qlearning_balancer"""
    seed = run_id
    rng_py = random.Random(seed)

    popularities = build_popularities(traffic_type, catalog_size, alpha, real_trace_path, seed)
    req = np.random.default_rng(seed).choice(catalog_size, num_requests, p=popularities)

    monitor = Monitor(num_servers, catalog_size, seed=seed)

    rows = []

    # --- Optimal_QLRU (cache.py) ---
    initial_state = rng_py.sample(range(catalog_size), min(capacity, catalog_size))
    sizes = [1] * catalog_size
    opt_cache = Optimal_QLRU(initial_state, capacity, beta, sizes, monitor.file_to_server)
    res_opt = monitor.run_metrics(opt_cache, req, popularities, capacity, algorithm="Optimal_QLRU", plot=False)
    rows.append({
        "algorithm": "Optimal_QLRU",
        "hit_rate": res_opt["hit_rate"],
        "jfi": res_opt["jfi"],
        "angle_degrees": res_opt["angle_degrees"],
    })

    # --- Q-learning + baselines ---
    for mode in QLEARN_MODES:
        res_q = run_qlearning(monitor, req, catalog_size, capacity, mode=mode, seed=seed)
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
    }
    for row in rows:
        row.update(common)
    return rows
