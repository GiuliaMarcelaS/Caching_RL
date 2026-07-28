"""
Preprocessa o Azure Functions Invocation Trace 2021 (CSV) para um parquet
compacto com colunas [content_id, timestamp], ordenado por timestamp.

Implementacao em duas passagens para nao estourar memoria em traces grandes:

  1a passagem (streaming): le o CSV em chunks, limpa/projeta cada chunk e
     grava imediatamente num parquet intermediario NAO ordenado. Nunca mais
     de ~--chunksize linhas ficam na memoria de uma vez.
  2a passagem: le o parquet intermediario inteiro (agora com so 2 colunas
     compactas, sem as colunas originais do CSV) e ordena por timestamp.

A ordenacao global exige o dataset inteiro em memoria em algum momento -- nao
tem como evitar isso mantendo a ordem exata sem um merge-sort externo em
disco. O ganho aqui e nao manter, ao mesmo tempo, a lista de chunks JUNTO com
a concatenacao inteira (que e o que a versao anterior fazia), reduzindo o
pico de memoria para aproximadamente 1x o tamanho do dataset final em vez de
2-3x.
"""
import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    usecols = ["app", "func", "end_timestamp"]
    tmp_path = args.output_parquet + ".unsorted.tmp"

    # --- 1a passagem: streaming, escreve incrementalmente ---
    writer = None
    n_rows = 0
    try:
        for i, chunk in enumerate(pd.read_csv(args.input_csv, usecols=usecols, chunksize=args.chunksize)):
            chunk = chunk.dropna(subset=["app", "func", "end_timestamp"])
            chunk["content_id"] = chunk["app"].astype(str) + ":" + chunk["func"].astype(str)
            chunk = chunk.rename(columns={"end_timestamp": "timestamp"})[["content_id", "timestamp"]]

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)

            n_rows += len(chunk)
            print(f"  chunk {i + 1} lido e gravado ({len(chunk):,} linhas, {n_rows:,} acumuladas)")
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        raise ValueError(
            "Nenhuma linha valida encontrada apos remover NaNs em app/func/end_timestamp. "
            "Verifique --input-csv."
        )

    # --- 2a passagem: le o intermediario compacto inteiro e ordena ---
    df = pd.read_parquet(tmp_path)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df.to_parquet(args.output_parquet, index=False)

    os.remove(tmp_path)

    n_distinct = df["content_id"].nunique()
    print(f"Total de eventos: {len(df):,}")
    print(f"Funcoes distintas: {n_distinct:,}")
    print(f"Salvo em: {args.output_parquet}")
    if n_distinct < 100_000:
        print("\nATENCAO: menos de 100.000 funcoes distintas reduza --catalog-size")
        print("nos scripts de experimento ou combine mais particoes do dataset.")


if __name__ == "__main__":
    main()