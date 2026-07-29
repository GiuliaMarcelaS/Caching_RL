"""
Plot experiment results: one figure per traffic type, two panels each
(Jain's Fairness Index and Hit Rate).

The plot type adapts to the data:
    1 distinct X value   -> horizontal BAR chart (a line plot would be empty)
    >1 distinct X values -> line plot with 95% confidence bands
    --dist               -> box plots of the individual runs, to reveal whether
                            a large std comes from pooling configurations
                            instead of run-to-run noise
    --split-by COL       -> one series per value of COL (e.g. capacity), so
                            different configurations are not averaged together
    
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed legend order, identical across every figure
LEGEND_ORDER = ["q-LRU", "LRU", "No Caching", "OPT", "Q-Learning"]

COLORS = {"q-LRU": "#7B3FA0", "LRU": "#E8871A", "No Caching": "#8A8A8A",
          "OPT": "#2E8B45", "Q-Learning": "#2F6FD0"}
MARKERS = {"q-LRU": "s", "LRU": "^", "No Caching": "v",
           "OPT": "X", "Q-Learning": "o"}

# How each traffic_type value is rendered in titles and file names
TRAFFIC_NAMES = {
    "irm": "IRM",
    "real": "Real Traffic",
    "real_traffic": "Real Traffic",
    "azure": "Real Traffic (Azure)",
    "trace": "Real Traffic",
}

REQUIRED_COLUMNS = ("traffic_type", "algorithm", "hit_rate", "jfi")

# Two-sided 95% t critical value by degrees of freedom (n - 1)
_T95 = {4: 2.776, 5: 2.571, 9: 2.262, 14: 2.145, 19: 2.093,
        24: 2.064, 29: 2.045, 49: 2.010, 99: 1.984}


def t_critical(dof):
    """95% two-sided t critical value; falls back to the normal quantile."""
    if dof <= 0:
        return 0.0
    for k in sorted(_T95):
        if dof <= k:
            return _T95[k]
    return 1.96


def get_academic_label(raw_alg_name: str) -> str:
    """Map the raw algorithm names in the CSV to the paper's labels."""
    raw = str(raw_alg_name).lower()
    if "optimal_qlru" in raw:
        return "q-LRU"
    elif "qlearning_balancer[qlearn]" in raw:
        return "Q-Learning"
    elif "qlearning_balancer[lru]" in raw:
        return "LRU"
    elif "qlearning_balancer[nocache]" in raw:
        return "No Caching"
    elif "qlearning_balancer[oracle]" in raw or "fairstatic" in raw:
        return "OPT"
    return str(raw_alg_name)  # fallback


def order_labels(labels):
    """Sort labels by LEGEND_ORDER, appending anything unrecognised."""
    labels = list(labels)
    return ([l for l in LEGEND_ORDER if l in labels]
            + [l for l in labels if l not in LEGEND_ORDER])


def traffic_title(traffic_type):
    return TRAFFIC_NAMES.get(str(traffic_type).lower(), str(traffic_type))


# load
def load(csv_paths):
    """Read one or more CSV files and concatenate them."""
    frames = []
    for path in csv_paths:
        if not os.path.exists(path):
            sys.exit(f"ERROR: file not found: {path}")
        frame = pd.read_csv(path)
        frame["_source"] = os.path.basename(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["alg"] = df["algorithm"].map(get_academic_label)
    return df


#  diagnose
def diagnose(df, group_col, split_by=None):
    """Report what the CSV actually contains before anything is plotted.

    Two failure modes are caught here:
      1. only one traffic_type present  -> only one figure is produced
      2. more rows per point than seeds -> configurations are being pooled,
         which makes the error bars meaningless
    """
    print("=" * 74)
    print("CSV DIAGNOSTICS")
    print("=" * 74)

    for col in REQUIRED_COLUMNS + (group_col,):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' is missing from the CSV.\n"
                     f"Columns found: {list(df.columns)}")
    if split_by and split_by not in df.columns:
        sys.exit(f"ERROR: --split-by column '{split_by}' is missing from the CSV.\n"
                 f"Columns found: {list(df.columns)}")

    n_nan = df["traffic_type"].isna().sum()
    if n_nan:
        print(f"WARNING: {n_nan} rows have an empty traffic_type. groupby would "
              f"drop them silently. Fill them in or remove them.")

    print(f"\ntotal rows: {len(df)}")
    print("\nrows per traffic_type:")
    print(df["traffic_type"].value_counts(dropna=False).to_string())

    x_values = sorted(df[group_col].dropna().unique())
    print(f"\ndistinct {group_col} values: {x_values}")
    if len(x_values) == 1:
        print(f"  -> only ONE point on the X axis. A line plot would be empty, "
              f"so a bar chart is used instead.")

    print(f"\nruns per (algorithm, {group_col}):")
    table = df.pivot_table(index="algorithm", columns=group_col,
                           values="hit_rate", aggfunc="size", fill_value=0)
    print(table.to_string())

    # --- only one traffic type -------------------------------------------
    traffic_types = df["traffic_type"].dropna().unique()
    if len(traffic_types) < 2:
        print("\n" + "!" * 74)
        print(f"ONLY ONE traffic_type IN THE CSV: {list(traffic_types)}")
        print("That is why a single figure is produced. The script is fine -- "
              "the data is missing.")
        print("Check whether the real-traffic runs:")
        print("  (a) have actually been executed;")
        print("  (b) were written to a DIFFERENT csv -> pass both files:")
        print("      python plot_results.py exp1_irm.csv exp1_real.csv")
        print("  (c) recorded traffic_type under another name, or left it empty.")
        print("!" * 74)

    # --- configurations pooled into one point 
    if "seed" in df.columns and len(table.values):
        n_seeds = df["seed"].nunique()
        max_per_point = int(table.values.max())
        if max_per_point > n_seeds:
            factor = max_per_point / n_seeds
            print("\n" + "!" * 74)
            print(f"AGGREGATION ALERT: {max_per_point} rows per point, but only "
                  f"{n_seeds} distinct seeds ({factor:.0f}x).")
            print("Each point is averaging runs from DIFFERENT CONFIGURATIONS.")
            varying = [c for c in df.columns
                       if c not in (group_col, "seed", "algorithm", "alg",
                                    "traffic_type", "hit_rate", "jfi", "_source")
                       and df[c].nunique() > 1]
            if varying:
                print(f"Columns still varying inside a point: {varying}")
                print("Use --split-by <column>, or filter the CSV before plotting.")
            print("The standard deviation here is NOT run-to-run noise -- it is "
                  "variation across configurations.")
            print("The error bars are meaningless until this is fixed.")
            print("!" * 74)

    # implausible dispersion 
    grp = df.groupby(["algorithm", group_col])["hit_rate"]
    cv = (grp.std() / grp.mean().replace(0, np.nan)).dropna()
    if len(cv) and cv.max() > 0.25:
        print(f"\nWARNING: hit_rate coefficient of variation reaches {cv.max():.2f} "
              f"(expected < 0.05 with 10M requests and common random numbers).")

    print("=" * 74 + "\n")


# aggregate
def summarize(df, keys):
    """Mean, std and 95% CI half-width for both metrics."""
    agg = df.groupby(keys).agg(
        hit_rate_mean=("hit_rate", "mean"),
        hit_rate_std=("hit_rate", "std"),
        jfi_mean=("jfi", "mean"),
        jfi_std=("jfi", "std"),
        n=("hit_rate", "size"),
    ).reset_index()
    tc = agg["n"].apply(lambda n: t_critical(int(n) - 1))
    agg["hit_ci"] = tc * agg["hit_rate_std"].fillna(0) / np.sqrt(agg["n"])
    agg["jfi_ci"] = tc * agg["jfi_std"].fillna(0) / np.sqrt(agg["n"])
    return agg


# ------------------------------------------------------------- bar figure
def figure_bars(d, traffic_type, group_col, base_out, fmt):
    """Used when the sweep has a single X value."""
    x_value = d[group_col].iloc[0]
    labels = order_labels(d["alg"].unique())
    d = d.set_index("alg").loc[labels].reset_index()
    y = np.arange(len(labels))[::-1]

    x_axis = "Number of Servers" if group_col == "num_servers" else "Cache Capacity"
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.75 * len(labels) + 2.6))

    panels = [("jfi_mean", "jfi_ci", f"Fairness vs. {x_axis}",
               "Jain's Fairness Index", 1.0),
              ("hit_rate_mean", "hit_ci", f"Hit Rate vs. {x_axis}",
               "Hit Rate (%)", 100.0)]

    for i, (mean_col, ci_col, title, xlabel, scale) in enumerate(panels):
        values = d[mean_col].values * scale
        errors = d[ci_col].values * scale
        ax[i].barh(y, values, xerr=errors,
                   color=[COLORS.get(l, "#555") for l in labels], alpha=.85,
                   error_kw=dict(ecolor="black", capsize=4, lw=1.2), height=.62)
        span = max(values) if len(values) and max(values) else 1.0
        for yy, v, e in zip(y, values, errors):
            ax[i].text(v + e + 0.012 * span, yy,
                       f"{v:.3f}" if scale == 1.0 else f"{v:.1f}%",
                       va="center", fontsize=9)
        ax[i].set_yticks(y)
        ax[i].set_yticklabels(labels)
        ax[i].set_xlabel(xlabel)
        ax[i].set_title(title)
        ax[i].grid(alpha=.3, axis="x")
        ax[i].set_axisbelow(True)
        ax[i].set_xlim(0, span * 1.22)
        ax[i].set_ylim(-1.05, len(labels) - .3)

    # reference lines on the fairness panel
    ax[0].axvline(1.0, color="k", ls="--", lw=1, alpha=.6)
    ax[0].text(1.0, -0.72, "maximum (1.0)", fontsize=8, alpha=.75,
               ha="center", va="top")
    if group_col == "num_servers" and float(x_value) > 0:
        floor = 1.0 / float(x_value)
        ax[0].axvline(floor, color="r", ls=":", lw=1, alpha=.6)
        ax[0].text(floor, -0.72, f"worst case (1/N = {floor:.2f})", fontsize=8,
                   color="r", alpha=.85, ha="center", va="top")

    fig.suptitle(f"{traffic_title(traffic_type)} — {group_col} = {x_value:g}",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = f"{base_out}_{traffic_type}_bars.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


#  line figure
def axis_label(col):
    return {"num_servers": "Number of Servers",
            "capacity": "Cache Capacity"}.get(col, col)


def figure_lines(d, traffic_type, x_col, base_out, fmt, ci=False, split_by=None):
    x_axis = axis_label(x_col)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    panels = [("jfi_mean", "jfi_ci", "Jain's Fairness Index",
               f"Fairness vs. {x_axis}", 1.0),
              ("hit_rate_mean", "hit_ci", "Hit Rate (%)",
               f"Hit Rate vs. {x_axis}", 100.0)]

    for i, (mean_col, ci_col, ylabel, title, scale) in enumerate(panels):
        for label in order_labels(d["alg"].unique()):
            sub = d[d["alg"] == label]
            groups = [(None, sub)] if split_by is None else list(sub.groupby(split_by))
            for k, (key, g) in enumerate(groups):
                g = g.sort_values(x_col)
                name = label if key is None else f"{label} ({split_by}={key:g})"
                ax[i].plot(g[x_col], g[mean_col] * scale,
                           marker=MARKERS.get(label, "o"), color=COLORS.get(label),
                           ls=["-", "--", ":", "-."][k % 4], label=name,
                           markersize=6, lw=1.8)
                if ci:
                    ax[i].fill_between(g[x_col],
                                       (g[mean_col] - g[ci_col]) * scale,
                                       (g[mean_col] + g[ci_col]) * scale,
                                       color=COLORS.get(label), alpha=.15, lw=0)
        ax[i].set_xscale("log")
        ax[i].set_xlabel(x_axis)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(title)
        ax[i].legend(fontsize=8)
        ax[i].grid(alpha=.3)

    fig.suptitle(traffic_title(traffic_type), fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{base_out}_{traffic_type}.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# distribution figure
def figure_distribution(df, traffic_type, base_out, fmt):
    """Box plots of the individual runs.
    """
    d = df[df["traffic_type"] == traffic_type]
    labels = order_labels(d["alg"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    rng = np.random.default_rng(0)

    for i, (col, ylabel, scale) in enumerate([("jfi", "Jain's Fairness Index", 1.0),
                                              ("hit_rate", "Hit Rate (%)", 100.0)]):
        data = [d[d["alg"] == l][col].values * scale for l in labels]
        style = dict(patch_artist=True, widths=.55,
                     medianprops=dict(color="black", lw=1.5),
                     flierprops=dict(marker=".", ms=3, alpha=.4))
        try:                                   
            bp = ax[i].boxplot(data, tick_labels=labels, **style)
        except TypeError:                       
            bp = ax[i].boxplot(data, labels=labels, **style)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(COLORS.get(label, "#777"))
            patch.set_alpha(.55)
        for j, values in enumerate(data, start=1):
            ax[i].scatter(rng.normal(j, .06, len(values)), values,
                          s=5, color="black", alpha=.25, zorder=3)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(f"Distribution — {ylabel}")
        ax[i].grid(alpha=.3, axis="y")
        ax[i].set_axisbelow(True)
        ax[i].tick_params(axis="x", rotation=20)

    fig.suptitle(f"{traffic_title(traffic_type)} — distribution of individual runs",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = f"{base_out}_{traffic_type}_dist.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


#  driver
def plot(csv_paths, group_col, out_path=None, ci=False, fmt="png",
         split_by=None, dist=False):
    df = load(csv_paths)
    diagnose(df, group_col, split_by)

    base_out = (os.path.splitext(out_path)[0] if out_path
                else os.path.splitext(csv_paths[0])[0])

    keys = ["traffic_type", "alg", group_col] + ([split_by] if split_by else [])
    agg = summarize(df, keys)

    # Pick the X axis:
    #   sweep on group_col            -> lines on group_col (split_by = extra series)
    #   single group_col but split_by -> lines on split_by (that IS the sweep)
    #   single group_col, no split_by -> bar chart
    n_x = df[group_col].nunique()
    if n_x > 1:
        x_col, series_col = group_col, split_by
    elif split_by and df[split_by].nunique() > 1:
        x_col, series_col = split_by, None
        print(f"NOTE: {group_col} has a single value; using '{split_by}' as the "
              f"X axis instead.\n")
    else:
        x_col, series_col = None, None

    produced = []
    for traffic_type in sorted(df["traffic_type"].dropna().unique()):
        d = agg[agg["traffic_type"] == traffic_type]
        if x_col is None:
            out = figure_bars(d, traffic_type, group_col, base_out, fmt)
        else:
            out = figure_lines(d, traffic_type, x_col, base_out, fmt,
                               ci, series_col)
        produced.append(out)
        print(f"[OK] figure ({traffic_title(traffic_type)}) saved to {out}")
        if dist:
            out = figure_distribution(df, traffic_type, base_out, fmt)
            produced.append(out)
            print(f"[OK] distribution figure saved to {out}")

    n_main = len([p for p in produced if not p.endswith(f"_dist.{fmt}")])
    if n_main < 2:
        print(f"\nWARNING: one figure per traffic type was expected (IRM + real). "
              f"Only {n_main} was produced. See the diagnostics above.")
    return produced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="+",
                        help="one or more CSV files (they are concatenated)")
    parser.add_argument("--group-col", default="num_servers",
                        help="num_servers (exp1) or capacity (exp2)")
    parser.add_argument("--split-by", default=None,
                        help="column that still varies within a point, "
                             "e.g. capacity; draws one series per value")
    parser.add_argument("--out", default=None,
                        help="output file prefix")
    parser.add_argument("--ci", action="store_true",
                        help="draw 95%% confidence bands around the lines")
    parser.add_argument("--dist", action="store_true",
                        help="also produce box plots of the individual runs")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf"],
                        help="pdf is vectorial, better for the paper")
    args = parser.parse_args()
    plot(args.csv_path, args.group_col, args.out, args.ci, args.fmt,
         args.split_by, args.dist)