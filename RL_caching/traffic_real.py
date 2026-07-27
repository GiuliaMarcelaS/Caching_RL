"""
Trafego real: Azure Functions Invocation Trace 2021
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md

"""
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=4)
def _load_trace(trace_parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(trace_parquet_path, columns=["content_id", "timestamp"])


def empirical_popularity_from_trace(trace_parquet_path: str, catalog_size: int,
                                     seed: int = 0) -> np.ndarray:
    df = _load_trace(trace_parquet_path)
    counts = df["content_id"].value_counts()  # frequencia real de invocacao por funcao
    all_ids = counts.index.to_numpy()

    if catalog_size > len(all_ids):
        raise ValueError(
            f"catalog_size={catalog_size} bigger than number of distinct content_ids "
            f"no trace ({len(all_ids)}). Reduce the catalog_size or use one trace bigger."
        )

    rng = np.random.default_rng(seed)
    sampled_ids = rng.choice(all_ids, size=catalog_size, replace=False)
    sampled_counts = counts.loc[sampled_ids].to_numpy(dtype=np.float64)

    total = sampled_counts.sum()
    if total <= 0:
        raise ValueError("Sum of sampled counts is zero, cannot compute popularity distribution.")

    return sampled_counts / total
