"""
Gera graficos (JFI e hit rate vs. group-col) a partir dos CSVs produzidos por
run_experiment1.py / run_experiment2.py.

Produz um par de graficos SEPARADO por traffic_type (irm, real, ...) -- nunca
mistura os dois no mesmo eixo. Se o(s) CSV(s) de entrada tiverem so um
traffic_type (o caso comum, ja que run_experiment*.py agora grava irm/real em
arquivos separados por padrao), gera so um par de graficos; se tiverem mais
de um (ex.: um CSV antigo gerado com --traffic both), gera um par para cada.

Alem do PNG, imprime no console e salva em CSV uma tabela numerica com
media/desvio padrao de hit_rate e jfi por algoritmo e valor de group-col --
para nao depender so de "olhar as bolinhas" do grafico.

Exemplos:
    # um CSV so (caso comum, gerado com --traffic irm ou --traffic real)
    python3 plot_results.py results/experiment1_scalability_irm.csv

    # combinando dois CSVs (rodados em comandos separados) numa mesma leva de graficos
    python3 plot_results.py results/experiment1_scalability_irm.csv results/experiment1_scalability_real.csv
"""
import argparse
import os

import pandas as pd

# Rotulos de legenda/tabela, na ordem em que devem aparecer.
# ATENCAO: Optimal_QLRU -> "q-LRU" e qlearning_balancer[oracle] -> "OPT" e uma
# escolha de nomenclatura (ver conversa) -- se estiver invertido pro seu
# artigo, troque as duas linhas correspondentes abaixo.
ALGO_LABELS = {
    "Optimal_QLRU": "q-LRU",
    "qlearning_balancer[lru]": "LRU",
    "qlearning_balancer[nocache]": "No Caching",
    "qlearning_balancer[oracle]": "OPT",
    "qlearning_balancer[qlearn]": "Q-learning",
}
LABEL_ORDER = ["q-LRU", "LRU", "No Caching", "OPT", "Q-learning"]


def _label_for(algorithm: str) -> str:
    return ALGO_LABELS.get(algorithm, algorithm)


def _summary_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = df.groupby(["algorithm", group_col]).agg(
        hit_rate_mean=("hit_rate", "mean"),
        hit_rate_std=("hit_rate", "std"),
        jfi_mean=("jfi", "mean"),
        jfi_std=("jfi", "std"),
        n_runs=("jfi", "count"),
    ).reset_index()
    table["label"] = table["algorithm"].map(_label_for)
    table["hit_rate_mean_%"] = (table["hit_rate_mean"] * 100).round(2)
    table["hit_rate_std_%"] = (table["hit_rate_std"] * 100).round(2)
    table["jfi_mean"] = table["jfi_mean"].round(4)
    table["jfi_std"] = table["jfi_std"].round(4)

    table["label"] = pd.Categorical(table["label"], categories=LABEL_ORDER, ordered=True)
    table = table.sort_values([group_col, "label"])

    return table[["label", group_col, "hit_rate_mean_%", "hit_rate_std_%",
                   "jfi_mean", "jfi_std", "n_runs"]].rename(columns={"label": "algorithm"})


def _plot_one_traffic(df: pd.DataFrame, traffic_type: str, group_col: str,
                       out_png: str, out_summary_csv: str, show: bool):
    import matplotlib.pyplot as plt

    summary = df.groupby(["algorithm", group_col]).agg(
        hit_rate_mean=("hit_rate", "mean"),
        jfi_mean=("jfi", "mean"),
    ).reset_index()
    summary["label"] = summary["algorithm"].map(_label_for)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for label in LABEL_ORDER:
        g = summary[summary["label"] == label].sort_values(group_col)
        if g.empty:
            continue
        ax[0].plot(g[group_col], g["jfi_mean"], "o-", label=label)
        ax[1].plot(g[group_col], g["hit_rate_mean"] * 100, "o-", label=label)
    # qualquer algoritmo sem rotulo mapeado (nao deveria acontecer, mas evita
    # sumir dado silenciosamente se um novo modo for adicionado no futuro)
    outros = sorted(set(summary["algorithm"]) - set(ALGO_LABELS))
    for algorithm in outros:
        g = summary[summary["algorithm"] == algorithm].sort_values(group_col)
        ax[0].plot(g[group_col], g["jfi_mean"], "o--", label=f"{algorithm} (sem rotulo)")
        ax[1].plot(g[group_col], g["hit_rate_mean"] * 100, "o--", label=f"{algorithm} (sem rotulo)")

    ax[0].set_xscale("log")
    ax[0].set_xlabel(group_col)
    ax[0].set_ylabel("Jain's Fairness Index")
    ax[0].set_title(f"Balancing vs. {group_col} ({traffic_type})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].set_xscale("log")
    ax[1].set_xlabel(group_col)
    ax[1].set_ylabel("hit rate (%)")
    ax[1].set_title(f"Hit rate vs. {group_col} ({traffic_type})")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[{traffic_type}] grafico salvo em {out_png}")

    table = _summary_table(df, group_col)
    table.to_csv(out_summary_csv, index=False)
    print(f"[{traffic_type}] tabela numerica salva em {out_summary_csv}")
    print(f"\n=== resultado numerico ({traffic_type}) ===")
    print(table.to_string(index=False))
    print()

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot(csv_paths, group_col: str, out_dir: str = None, prefix: str = None, show: bool = False):
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    if out_dir is None:
        out_dir = os.path.dirname(csv_paths[0]) or "."
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(csv_paths[0]))[0]
    os.makedirs(out_dir, exist_ok=True)

    # backend precisa ser escolhido antes de importar pyplot / criar figuras
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")

    traffic_types = sorted(df["traffic_type"].unique())

    # evita nomes tipo "..._irm_irm.png" quando o CSV de entrada ja tem o
    # traffic_type no nome (padrao gerado por run_experiment1/2.py)
    for tt in traffic_types:
        if prefix.endswith(f"_{tt}"):
            prefix = prefix[: -len(f"_{tt}")]
            break

    for traffic_type in traffic_types:
        sub = df[df["traffic_type"] == traffic_type]
        out_png = os.path.join(out_dir, f"{prefix}_{traffic_type}.png")
        out_summary_csv = os.path.join(out_dir, f"{prefix}_{traffic_type}_summary.csv")
        _plot_one_traffic(sub, traffic_type, group_col, out_png, out_summary_csv, show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+", help="um ou mais CSVs de resultado (ex.: irm e real juntos)")
    parser.add_argument("--group-col", default="num_servers", help="num_servers (exp1) ou capacity (exp2)")
    parser.add_argument("--out-dir", default=None,
                         help="diretorio de saida para PNGs e tabelas (default: mesmo diretorio do "
                              "primeiro CSV de entrada)")
    parser.add_argument("--prefix", default=None,
                         help="prefixo dos arquivos de saida (default: nome do primeiro CSV sem "
                              "extensao). Sera gerado um <prefix>_<traffic_type>.png e "
                              "<prefix>_<traffic_type>_summary.csv para cada traffic_type presente.")
    parser.add_argument("--show", action="store_true", help="tambem abre uma janela interativa com os graficos")
    args = parser.parse_args()
    plot(args.csv_paths, args.group_col, args.out_dir, args.prefix, args.show)