import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import run_one_config

CACHE_CAPACITIES = [10, 30, 100, 300, 1000, 3000, 10000, 30000]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-servers", type=int, required=True,
                         help="melhor quantidade de servidores obtida no Experimento 1")
    parser.add_argument("--catalog-size", type=int, default=424)
    parser.add_argument("--num-requests", type=int, default=1000000)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--traffic", choices=["irm", "real", "both"], default="both")
    parser.add_argument("--real-trace-path", type=str, default="data/azure_trace_processed.parquet")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=2.3)
    parser.add_argument("--skew-fraction", type=float, default=0.0,
                         help="fracao do catalogo sempre fixa no servidor 0 (ver Monitor.__init__ em "
                              "monitor.py). Default 0.0 = distribuicao totalmente aleatoria entre os "
                              "servidores. Use 0.10 para reproduzir o comportamento historico (hotspot "
                              "fixo no servidor 0), por exemplo para comparar 'com hotspot' vs 'sem hotspot'.")
    parser.add_argument("--skew-mode", choices=["index", "exclude_top_k"], default="index",
                         help="index (default) = bloco fixo por indice de arquivo (equivale a aleatorio, "
                              "ja que popularidades sao embaralhadas). exclude_top_k = servidor 0 fica com "
                              "um bloco popular o bastante pra ficar sobrecarregado, mas EXCLUINDO os "
                              "--exclude-top-k mais populares do catalogo -- cria cenario onde cachear os "
                              "arquivos mais populares (otimo pra hit rate) nao ajuda o servidor 0 (otimo "
                              "pra fairness exige cachear conteudo especifico dele, que nao e o mais "
                              "popular). So tem efeito com --skew-fraction > 0.")
    parser.add_argument("--exclude-top-k", type=int, default=None,
                         help="usado so com --skew-mode exclude_top_k. Quantos dos arquivos mais populares "
                              "do catalogo ficam de fora do servidor 0. Default = capacity de cada job "
                              "(varia, ja que este experimento varia capacity).")
    parser.add_argument("--workload-mode", choices=["uniform", "independent"], default="uniform",
                         help="uniform (default) = todo arquivo pesa 1 na carga do servidor (popularidade "
                              "= carga, comportamento historico). independent = custo por arquivo sorteado "
                              "de uma log-normal desacoplada da popularidade (ver "
                              "req_generator.independent_workload).")
    parser.add_argument("--workload-sigma", type=float, default=1.0,
                         help="dispersao da log-normal usada em --workload-mode independent (ignorado em "
                              "uniform). sigma maior = pesos mais desiguais entre arquivos.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=str, default=None,
                         help="default = results/experiment2_capacity_<traffic>.csv (ex.: ..._irm.csv, "
                              "..._real.csv), ou results/experiment2_capacity.csv se --traffic both. "
                              "Assim, rodar irm e real em comandos separados nao sobrescreve um "
                              "arquivo no outro.")
    args = parser.parse_args()

    if args.out is None:
        suffix = "" if args.traffic == "both" else f"_{args.traffic}"
        args.out = f"results/experiment2_capacity{suffix}.csv"

    traffics = ["irm", "real"] if args.traffic == "both" else [args.traffic]

    if "real" in traffics and not os.path.isfile(args.real_trace_path):
        parser.error(
            f"--real-trace-path aponta para um arquivo inexistente: {args.real_trace_path!r}. "
            f"Rode preprocess_azure_trace.py primeiro para gera-lo, ou passe --traffic irm "
            f"para rodar so com trafego sintetico."
        )

    jobs = list(itertools.product(traffics, CACHE_CAPACITIES, range(args.num_runs)))
    print(f"servidores fixos={args.num_servers} | total de execucoes: {len(jobs)} (workers={args.workers})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None
    t0 = time.time()
    done = 0

    with open(args.out, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one_config, tt, args.num_servers, args.catalog_size, args.num_requests, cap,
                      run_id, args.alpha, args.beta, args.real_trace_path, args.skew_fraction,
                      args.skew_mode, args.exclude_top_k, args.workload_mode, args.workload_sigma): (tt, cap, run_id)
            for tt, cap, run_id in jobs
        }
        for fut in as_completed(futures):
            tt, cap, run_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(f"[ERRO] traffic={tt} capacity={cap} run={run_id}: {exc}")
                continue

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
            f.flush()

            done += 1
            elapsed = time.time() - t0
            resumo = ", ".join(f"{r['algorithm']}={r['jfi']:.3f}" for r in rows)
            print(f"[{done}/{len(jobs)}] traffic={tt} capacity={cap} run={run_id} jfi[{resumo}] ({elapsed:.1f}s)")

    print(f"Concluido em {time.time() - t0:.1f}s. Resultados em {args.out}")


if __name__ == "__main__":
    main()