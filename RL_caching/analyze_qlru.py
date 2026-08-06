import os
import random
import numpy as np
import pandas as pd

from cache import Optimal_QLRU
from monitor import Monitor
from q_learning import run_qlearning
from req_generator import zipf

SEED = 0
NUM_SERVERS = 2
CATALOG_SIZE = 100_000
NUM_REQUESTS = 1000
CAPACITY = 1_000
ALPHA = 0.8
BETA = 2.3

popularities = zipf(ALPHA, CATALOG_SIZE, seed=SEED)
req = np.random.default_rng(SEED).choice(CATALOG_SIZE, NUM_REQUESTS, p=popularities)
monitor = Monitor(NUM_SERVERS, CATALOG_SIZE, seed=SEED, skew_fraction=0.0)
sizes = [1] * CATALOG_SIZE

# --- Optimal_QLRU com rastreamento ligado ---
opt_cache = Optimal_QLRU([], CAPACITY, BETA, sizes, monitor.file_to_server,
                          rng=random.Random(SEED), track_stats=True)
res_opt = monitor.run_metrics(opt_cache, req, popularities, CAPACITY, algorithm="Optimal_QLRU",
                               plot=False, curve_points=200)

q_hist = np.array(opt_cache.q_history)
delta_hist = np.array(opt_cache.delta_jfi_history)

print("=== ESTATISTICAS DE q_f (Optimal_QLRU) ===")
print(f"decisoes de admissao avaliadas (misses): {len(q_hist):,}")
print(f"media:            {q_hist.mean():.6f}")
print(f"mediana:          {np.median(q_hist):.6f}")
print(f"desvio padrao:    {q_hist.std():.6f}")
print(f"minimo / maximo:  {q_hist.min():.6f} / {q_hist.max():.6f}")
print(f"percentil 25/75:  {np.percentile(q_hist,25):.6f} / {np.percentile(q_hist,75):.6f}")
print(f"fracao com q_f=0 (delta_jfi<=0, nunca cacheia): {(q_hist == 0).mean()*100:.2f}%")
print(f"fracao com q_f>0.5:                             {(q_hist > 0.5).mean()*100:.2f}%")
print()
print("=== ESTATISTICAS DE delta_jfi (impacto marginal no JFI) ===")
print(f"media:            {delta_hist.mean():.8f}")
print(f"mediana:          {np.median(delta_hist):.8f}")
print(f"desvio padrao:    {delta_hist.std():.8f}")
print(f"fracao positiva (ajuda o balanceamento): {(delta_hist > 0).mean()*100:.2f}%")
print()
print(f"resultado final: hit_rate={res_opt['hit_rate']:.4f}  jfi={res_opt['jfi']:.4f}")

# --- Q-learning, para comparar convergencia ---
monitor2 = Monitor(NUM_SERVERS, CATALOG_SIZE, seed=SEED, skew_fraction=0.0)
res_q = run_qlearning(monitor2, req, CATALOG_SIZE, CAPACITY, mode="qlearn", seed=SEED, curve_points=200)
print(f"Q-learning: hit_rate={res_q['hit_rate']:.4f}  jfi={res_q['jfi']:.4f}")

# pasta de saida local (funciona em Windows/Linux/Mac -- nao usa /tmp)
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# salva os dados brutos
df_opt_curve = pd.DataFrame(res_opt["jfi_curve"], columns=["step", "jfi"])
df_opt_curve["algorithm"] = "Optimal_QLRU"
df_q_curve = pd.DataFrame({"step": np.linspace(0, NUM_REQUESTS, len(res_q["curve"])).astype(int),
                            "jfi": res_q["curve"]})
df_q_curve["algorithm"] = "Q-learning"
df_curves = pd.concat([df_opt_curve, df_q_curve], ignore_index=True)
curves_path = os.path.join(out_dir, "jfi_convergence.csv")
df_curves.to_csv(curves_path, index=False)

df_q_stats = pd.DataFrame({"q_f": q_hist, "delta_jfi": delta_hist})
stats_path = os.path.join(out_dir, "q_f_stats.csv")
df_q_stats.to_csv(stats_path, index=False)
print()
print(f"dados salvos em {curves_path} e {stats_path}")

# --- grafico (convergencia do JFI + distribuicao de q_f) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

for alg, g in df_curves.groupby("algorithm"):
    g = g.sort_values("step")
    ax[0].plot(g["step"], g["jfi"], "-", label=alg, linewidth=1.5)
ax[0].set_xlabel("numero de requisicoes processadas")
ax[0].set_ylabel("Jain's Fairness Index")
ax[0].set_title("Convergencia do JFI ao longo do tempo")
ax[0].legend()
ax[0].grid(alpha=.3)

ax[1].hist(q_hist, bins=50, log=True)
ax[1].set_xlabel("q_f calculado")
ax[1].set_ylabel("frequencia (escala log)")
ax[1].set_title("Distribuicao de q_f (Optimal_QLRU)")
ax[1].grid(alpha=.3)

plt.tight_layout()
png_path = os.path.join(out_dir, "qlru_analysis.png")
plt.savefig(png_path, dpi=150)
print(f"grafico salvo em {png_path}")