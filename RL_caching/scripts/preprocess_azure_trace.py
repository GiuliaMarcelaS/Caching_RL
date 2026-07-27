import argparse 
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    usecols = ["app", "func", "end_timestamp"]
    chunks = []
    for i, chunk in enumerate(pd.read_csv(args.input_csv, usecols=usecols, chunksize=args.chunksize)):
        chunk = chunk.dropna(subset=["app", "func", "end_timestamp"])
        chunk["content_id"] = chunk["app"].astype(str) + ":" + chunk["func"].astype(str)
        chunk = chunk.rename(columns={"end_timestamp": "timestamp"})[["content_id", "timestamp"]]
        chunks.append(chunk)
        print(f"  chunk {i + 1} lido ({len(chunk):,} linhas)")

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(args.output_parquet, index=False)

    n_distinct = df["content_id"].nunique()
    print(f"Total de eventos: {len(df):,}")
    print(f"Funcoes distintas: {n_distinct:,}")
    print(f"Salvo em: {args.output_parquet}")
    if n_distinct < 100_000:
        print("\nATENCAO: menos de 100.000 funcoes distintas reduza --catalog-size")
        print("nos scripts de experimento ou combine mais particoes do dataset.")


if __name__ == "__main__":
    main()
