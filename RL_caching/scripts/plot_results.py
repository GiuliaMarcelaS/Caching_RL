
import argparse
import os

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: str, group_col: str, out_path: str = None, show: bool = False):
    df = pd.read_csv(csv_path)
    summary = df.groupby(["traffic_type", "algorithm", group_col]).agg(
        hit_rate_mean=("hit_rate", "mean"),
        jfi_mean=("jfi", "mean"),
    ).reset_index()

    multi_traffic = summary["traffic_type"].nunique() > 1

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (traffic_type, alg), g in summary.groupby(["traffic_type", "algorithm"]):
        g = g.sort_values(group_col)
        label = f"{alg} ({traffic_type})" if multi_traffic else alg
        ax[0].plot(g[group_col], g["jfi_mean"], "o-", label=label)
        ax[1].plot(g[group_col], g["hit_rate_mean"] * 100, "o-", label=label)

    ax[0].set_xscale("log")
    ax[0].set_xlabel(group_col)
    ax[0].set_ylabel("Jain's Fairness Index")
    ax[0].set_title(f"Balancing vs. {group_col}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)

    ax[1].set_xscale("log")
    ax[1].set_xlabel(group_col)
    ax[1].set_ylabel("hit rate (%)")
    ax[1].set_title(f"Hit rate vs. {group_col}")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(out_path, dpi=150)
    print(f"Grafico salvo em {out_path}")

    if show:
        # troca pra um backend com janela e reexibe (Agg nao tem GUI)
        import matplotlib.pyplot as plt2
        plt2.switch_backend("TkAgg")
        plt2.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("--group-col", default="num_servers", help="num_servers (exp1) ou capacity (exp2)")
    parser.add_argument("--out", default=None, help="caminho do PNG a salvar (default: <csv_path sem extensao>.png)")
    parser.add_argument("--show", action="store_true", help="tambem abre uma janela interativa com o grafico")
    args = parser.parse_args()
    plot(args.csv_path, args.group_col, args.out, args.show)
