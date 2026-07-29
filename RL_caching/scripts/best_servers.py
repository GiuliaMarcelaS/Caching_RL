"""
Esse script usa a politica de cache do algoritmo de interesse (target) e compara com a politica de cache do baseline, para cada num_servers.
Ele reporta o gap de JFI (fairness) entre os dois algoritmos, e tambem o num_servers que produz a maior JFI absoluta para o algoritmo de interesse.
"""
import argparse
 
import pandas as pd
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("--target", default="Optimal_QLRU", help="algoritmo de interesse")
    parser.add_argument("--baseline", default="qlearning_balancer[nocache]", help="algoritmo de referencia")
    parser.add_argument("--traffic", default="irm", help="filtra por traffic_type (irm/real); default = mostra todos, separados")
    args = parser.parse_args()
 
    df = pd.read_csv(args.csv_path)
    if args.traffic:
        df = df[df["traffic_type"] == args.traffic]
 
    summary = df.groupby(["traffic_type", "algorithm", "num_servers"]).agg(
        jfi_mean=("jfi", "mean"),
        hit_rate_mean=("hit_rate", "mean"),
    ).reset_index()
 
    for traffic_type, g_traffic in summary.groupby("traffic_type"):
        pivot = g_traffic.pivot(index="num_servers", columns="algorithm", values="jfi_mean")
 
        if args.target not in pivot.columns or args.baseline not in pivot.columns:
            print(f"[{traffic_type}] algoritmo(s) nao encontrado(s). Disponiveis: {list(pivot.columns)}")
            continue
 
        gap = (pivot[args.target] - pivot[args.baseline]).sort_values(ascending=False)
 
        print(f"\n=== trafego: {traffic_type} ===")
        print(f"JFI de '{args.target}' menos JFI de '{args.baseline}', por num_servers "
              f"(maior = onde o algoritmo mais ajuda):")
        print(gap.to_string())
 
        best_gap_ns = int(gap.idxmax())
        best_jfi_ns = int(pivot[args.target].idxmax())
        print(f"\n--> maior VANTAGEM de '{args.target}' sobre '{args.baseline}': "
              f"num_servers = {best_gap_ns} (gap = {gap.max():.4f})")
        print(f"--> maior JFI absoluto de '{args.target}': num_servers = {best_jfi_ns} "
              f"(jfi = {pivot[args.target].max():.4f}) -- tende a ser sempre o menor num_servers "
              "testado; nao use isso sozinho como criterio.")
 
 
if __name__ == "__main__":
    main()