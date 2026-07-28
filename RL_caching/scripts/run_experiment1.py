import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import run_one_config

NUM_SERVIDORES = [2, 10, 20, 100, 200, 1000]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-size", type=int, default=100000)
    parser.add_argument("--num-requests", type=int, default=1000000)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--capacity", type=int, default=None,
                         help="default = 1%% do catalogo (igual ao notebook.ipynb original)")
    parser.add_argument("--traffic", choices=["irm", "real", "both"], default="both")
    parser.add_argument("--real-trace-path", type=str, default="data/azure_trace_processed.parquet")
    parser.add_argument("--alpha", type=float, default=0.8, help="expoente Zipf (IRM) -- default igual ao notebook")
    parser.add_argument("--beta", type=float, default=2.3, help="beta do Optimal_QLRU -- default igual ao notebook")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=str, default="results/experiment1_scalability.csv")
    args = parser.parse_args()

    capacity = args.capacity if args.capacity is not None else max(1, int(args.catalog_size * 0.01))
    traffics = ["irm", "real"] if args.traffic == "both" else [args.traffic]
    jobs = list(itertools.product(traffics, NUM_SERVIDORES, range(args.num_runs)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None
    t0 = time.time()
    done = 0

    with open(args.out, "w", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one_config, tt, ns, args.catalog_size, args.num_requests, capacity,
                      run_id, args.alpha, args.beta, args.real_trace_path): (tt, ns, run_id)
            for tt, ns, run_id in jobs
        }
        for fut in as_completed(futures):
            tt, ns, run_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(f"[ERRO] traffic={tt} servers={ns} run={run_id}: {exc}")
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
            print(f"[{done}/{len(jobs)}] traffic={tt} servers={ns} run={run_id} jfi[{resumo}] ({elapsed:.1f}s)")

    print(f"Concluido em {time.time() - t0:.1f}s. Resultados em {args.out}")


if __name__ == "__main__":
    main()
