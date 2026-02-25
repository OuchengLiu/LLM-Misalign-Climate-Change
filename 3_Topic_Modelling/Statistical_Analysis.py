from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def load_cfg() -> dict:
    """Load Configs.yaml from repo root and return the '3_Topic_Modelling' section."""
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "Configs.yaml").open("r", encoding="utf-8") as fp:
        root = yaml.safe_load(fp) or {}
    return root.get("3_Topic_Modelling", {})

CFG = load_cfg()

# Reuse 'subject' from Topic_Merge for default path building; this script-specific
# settings live under 'Topic_Dataset_Stats'.
MCFG = CFG.get("Topic_Merge", {})
STCFG = CFG.get("Statistical_Analysis", {})

SUBJECT = MCFG.get("subject", "Climate Change")
SUBJECT_SLUG = SUBJECT.replace(" ", "_")

# ---- Paths -------------------------------------------------------------------
BASE_DIR = Path(
    STCFG.get("base_dir", f"../Logs/Test/{SUBJECT_SLUG}")
).expanduser()

_data_file_cfg = STCFG.get("data_file", "")
_out_csv_cfg   = STCFG.get("out_csv", "")

DATA_FILE = Path(_data_file_cfg).expanduser() if _data_file_cfg else (BASE_DIR / "All_Data_with_Final_Topic.jsonl")
OUT_CSV   = Path(_out_csv_cfg).expanduser()   if _out_csv_cfg   else (BASE_DIR / "Topic_Dataset_Stats_FilterOut.csv")

# ---- Dataset list and matching regex -----------------------------------------
DEFAULT_DATASETS: List[str] = [
    "WildChat", "LMSYSChat", "Reddit", "ClimateQ&A", "ClimSight", "ClimaQA_Gold", "ClimaQA_Silver", "Climate_FEVER",
    "Environmental_Claims", "SciDCC", "IPCC_AR6"
]
DATASETS: List[str] = list(STCFG.get("datasets", DEFAULT_DATASETS))

_dataset_rx_str_cfg = STCFG.get("dataset_regex", "")
if _dataset_rx_str_cfg:
    DATASET_RX = re.compile(_dataset_rx_str_cfg)
else:
    union = "|".join(re.escape(ds) for ds in DATASETS)
    DATASET_RX = re.compile(rf"^({union})")

# ---- Other parameters ---------------------------------------------------------
MIN_SUM_DEFAULT: int = int(STCFG.get("min_sum_default", 30))
KDE_POINTS: int = int(STCFG.get("kde_points", 200))
PLOT_KDE: bool = bool(STCFG.get("plot_kde", True))
PLOT_BOXPLOT: bool = bool(STCFG.get("plot_boxplot", True))
PLOT_SIM_MATRIX: bool = bool(STCFG.get("plot_similarity_matrix", True))
PLOT_FORMAT: str = str(STCFG.get("plot_format_default", "png"))

RULES_DEFAULT: List[str] = list(STCFG.get("rules_default", ["global", "ds", "topn"]))
RULE_MODE_DEFAULT: str = str(STCFG.get("rule_mode_default", "any")).lower()

MIN_GLOBAL_PCT_DEFAULT: float = float(STCFG.get("min_global_pct_default", 0.002))
MIN_PCT_ANY_DEFAULT: float    = float(STCFG.get("min_pct_any_default", 0.01))
MIN_PCT_K_DEFAULT: int        = int(STCFG.get("min_pct_k_default", 2))
TOPN_DEFAULT: int             = int(STCFG.get("topn_default", 20))
TOPN_K_DEFAULT: int           = int(STCFG.get("topn_k_default", 1))

# --- UPDATED: similarity/plot percentage source from YAML ---------------------
# - 'filtered_pct'      -> kept topics, exclude <NONE> / Irrelevant Data      [B]
# - 'orig_no_none_pct'  -> original data, exclude <NONE> / Irrelevant Data    [A]
PLT_SOURCE_DEFAULT: str = str(STCFG.get("plt_source_default", "filtered_pct")).lower()

# -----------------------------------------------------------------------------
# Utility: Gini coefficient
# -----------------------------------------------------------------------------

def gini(arr: np.ndarray) -> float:
    """Compute Gini coefficient for a non-negative array."""
    a = arr.flatten()
    if np.any(a < 0):
        raise ValueError("Gini requires non-negative values")
    total = a.sum()
    if total == 0:
        return 0.0
    sorted_ = np.sort(a)
    n = len(a)
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * sorted_) / (n * total) - (n + 1) / n)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-dataset counts for each Final_Topic, two percentage variants, extra variance & Gini metrics, and draw KDE + boxplots + similarity matrix."
    )
    p.add_argument(
        "--min-sum", "-m",
        type=int,
        default=MIN_SUM_DEFAULT,
        help=f"LEGACY (unused in filtering): prior global count threshold (default from config: {MIN_SUM_DEFAULT})."
    )
    p.add_argument(
        "--rules",
        nargs="*",
        choices=["global", "ds", "topn"],
        default=RULES_DEFAULT,
        help="Which filtering rules to enable: 'global' (global share), 'ds' (per-dataset share in ≥K datasets), 'topn' (Top-N in ≥K datasets). Default from YAML."
    )
    p.add_argument(
        "--rule-mode",
        choices=["any", "all"],
        default=RULE_MODE_DEFAULT,
        help="How to combine enabled rules: 'any' (ANY enabled rule satisfied), 'all' (ALL enabled rules must be satisfied). Default from YAML."
    )
    p.add_argument("--min-global-pct", type=float, default=MIN_GLOBAL_PCT_DEFAULT,
                   help=f"Global share threshold (default from YAML or code).")
    p.add_argument("--min-pct-any", type=float, default=MIN_PCT_ANY_DEFAULT,
                   help=f"Per-dataset share threshold (default from YAML or code).")
    p.add_argument("--min-pct-k", type=int, default=MIN_PCT_K_DEFAULT,
                   help=f"Minimum number of datasets meeting per-dataset share (default from YAML or code).")
    p.add_argument("--topn", type=int, default=TOPN_DEFAULT,
                   help=f"Top-N rank per dataset (default from YAML or code).")
    p.add_argument("--topn-k", type=int, default=TOPN_K_DEFAULT,
                   help=f"Minimum number of datasets where topic is within Top-N (default from YAML or code).")
    p.add_argument(
        "--plot-format",
        type=str,
        default=PLOT_FORMAT,
        help="Output format for plots (e.g., png, svg, pdf). Default from YAML."
    )
    p.add_argument(
        "--plt-source",
        type=str,
        choices=["filtered_pct", "orig_no_none_pct"],
        default=PLT_SOURCE_DEFAULT,
        help="Which percentage baseline to use for KDE, boxplot, and similarity matrix. Default from YAML."
    )
    return p.parse_args()

# -----------------------------------------------------------------------------
# Cosine similarity helpers
# -----------------------------------------------------------------------------

def _cosine_similarity_matrix(columns: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    labels = [lbl for lbl, _ in columns]
    if not columns:
        return np.zeros((0, 0)), labels
    M = np.column_stack([v for _, v in columns])  # topics x datasets
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    M_norm = M / norms
    S = np.clip(M_norm.T @ M_norm, 0.0, 1.0)
    return S, labels

def _plot_similarity_matrix(S: np.ndarray, labels: List[str], out_png: Path) -> None:
    if S.size == 0:
        return
    n = len(labels)
    fig_size = max(6, min(1.0 * n + 2, 18))
    plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(S, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.xticks(ticks=np.arange(n), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(n), labels=labels)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.title("Cosine Similarity of Dataset Topic Distributions")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"✅ Saved dataset similarity matrix to {out_png}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    enabled_rules = set(args.rules)
    rule_mode = args.rule_mode
    plot_format = args.plot_format.lower()
    plt_source = args.plt_source.lower()

    if not DATA_FILE.exists():
        print(f"Error: input file {DATA_FILE} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Topics to ignore for percentage calculations & plotting
    IGNORE_PCT_TOPICS = {"<NONE>", "Irrelevant Data", "Irelevant Data", "F1. Others"}

    # Read and tally counts per topic per dataset
    original_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with DATA_FILE.open("r", encoding="utf-8") as fp:
        for ln in fp:
            row = json.loads(ln)
            m = DATASET_RX.match(row.get("id", ""))
            if not m:
                continue
            ds = m.group(1)
            topics = (
                row.get("Final_Topic")
                or row.get("Final_Topics")
                or row.get("Final_Topic_IDs")
                or ["<NONE>"]
            )
            if not isinstance(topics, (list, tuple)):
                topics = [topics]
            for tp in topics:
                original_counts[tp][ds] += 1

    # -----------------------------
    # Adaptive filtering conditions
    # -----------------------------
    min_global_pct = float(args.min_global_pct)
    min_pct_any    = float(args.min_pct_any)
    min_pct_k      = max(1, int(args.min_pct_k))
    topn           = max(1, int(args.topn))
    topn_k         = max(1, int(args.topn_k))

    ds_tot_incl_none = {ds: 0 for ds in DATASETS}
    ds_tot_excl_none = {ds: 0 for ds in DATASETS}
    for tp, by_ds in original_counts.items():
        for ds in DATASETS:
            c = by_ds.get(ds, 0)
            ds_tot_incl_none[ds] += c
            if tp != "<NONE>":
                ds_tot_excl_none[ds] += c

    global_tot_excl_none = sum(ds_tot_excl_none.values())
    global_tot = global_tot_excl_none if global_tot_excl_none > 0 else sum(ds_tot_incl_none.values())

    topn_sets: Dict[str, set] = {ds: set() for ds in DATASETS}
    for ds in DATASETS:
        pairs = [(tp, original_counts[tp].get(ds, 0)) for tp in original_counts.keys() if tp != "<NONE>"]
        pairs.sort(key=lambda x: x[1], reverse=True)
        if not pairs:
            continue
        if len(pairs) <= topn:
            cutoff = pairs[-1][1] if pairs else 0
        else:
            cutoff = pairs[topn - 1][1]
        for tp, c in pairs:
            if c > 0 and c >= cutoff:
                topn_sets[ds].add(tp)

    def _rule_global(tp: str, ds_counts: Dict[str, int]) -> bool:
        if "global" not in enabled_rules:
            return False
        total = sum(ds_counts.get(ds, 0) for ds in DATASETS)
        return (global_tot > 0) and ((total / global_tot) >= min_global_pct)

    def _rule_ds(tp: str, ds_counts: Dict[str, int]) -> bool:
        if "ds" not in enabled_rules:
            return False
        hits = 0
        for ds in DATASETS:
            den = ds_tot_excl_none[ds] if ds_tot_excl_none[ds] > 0 else ds_tot_incl_none[ds]
            if den == 0:
                continue
            if (ds_counts.get(ds, 0) / den) >= min_pct_any:
                hits += 1
                if hits >= min_pct_k:
                    return True
        return False

    def _rule_topn(tp: str) -> bool:
        if "topn" not in enabled_rules:
            return False
        top_hits = sum(1 for ds in DATASETS if tp in topn_sets[ds])
        return top_hits >= topn_k

    def _keep_topic(tp: str, ds_counts: Dict[str, int]) -> bool:
        r_global = _rule_global(tp, ds_counts)
        r_ds     = _rule_ds(tp, ds_counts)
        r_topn   = _rule_topn(tp)
        if rule_mode == "all":
            checks = []
            if "global" in enabled_rules: checks.append(r_global)
            if "ds" in enabled_rules:     checks.append(r_ds)
            if "topn" in enabled_rules:   checks.append(r_topn)
            return bool(checks) and all(checks)
        else:
            return r_global or r_ds or r_topn

    filtered_counts = {
        tp: ds_counts
        for tp, ds_counts in original_counts.items()
        if _keep_topic(tp, ds_counts)
    }
    removed = len(original_counts) - len(filtered_counts)
    if removed > 0:
        print(
            "Filtered out "
            f"{removed} topics using rules={sorted(enabled_rules)} mode={rule_mode} "
            f"(min_global_pct={min_global_pct}, min_pct_any={min_pct_any}, "
            f"min_pct_k={min_pct_k}, topn={topn}, topn_k={topn_k})"
        )

    # Column totals
    col_orig_all = {ds: sum(c.get(ds, 0) for c in original_counts.values()) for ds in DATASETS}

    # Exclude IGNORE_PCT_TOPICS for % baselines (A & B)
    col_orig_no_none = {
        ds: col_orig_all[ds] - sum(original_counts.get(t, {}).get(ds, 0) for t in IGNORE_PCT_TOPICS)
        for ds in DATASETS
    }
    col_filt_all = {ds: sum(c.get(ds, 0) for c in filtered_counts.values()) for ds in DATASETS}
    col_filt_no_none = {
        ds: col_filt_all[ds] - sum(filtered_counts.get(t, {}).get(ds, 0) for t in IGNORE_PCT_TOPICS)
        for ds in DATASETS
    }

    # Global total for B baseline percentages in CSV "Total" cell (exclude ignored topics)
    global_filt_total_no_ignored = sum(
        sum(ds_counts.get(ds, 0) for ds in DATASETS)
        for tp, ds_counts in filtered_counts.items()
        if tp not in IGNORE_PCT_TOPICS
    )

    # Matrices
    orig_all_mat = np.array([[original_counts[tp].get(ds, 0) for ds in DATASETS]
                             for tp in original_counts])
    orig_no_none_mat = np.array([[original_counts[tp].get(ds, 0) for ds in DATASETS]
                                 for tp in original_counts if tp not in IGNORE_PCT_TOPICS])
    filt_all_mat = np.array([[filtered_counts[tp].get(ds, 0) for ds in DATASETS]
                             for tp in filtered_counts])
    filt_no_none_mat = np.array([[filtered_counts[tp].get(ds, 0) for ds in DATASETS]
                                 for tp in filtered_counts if tp not in IGNORE_PCT_TOPICS])

    # ---------- Percentage dictionaries for two baselines (A / B) ----------
    pct_A_dict: Dict[str, np.ndarray] = {}
    if orig_no_none_mat.size:
        for i, ds in enumerate(DATASETS):
            den = col_orig_no_none[ds]
            pct_A_dict[ds] = (orig_no_none_mat[:, i] / den) if den > 0 else np.array([])
    else:
        pct_A_dict = {ds: np.array([]) for ds in DATASETS}

    pct_B_dict: Dict[str, np.ndarray] = {}
    if filt_no_none_mat.size:
        for i, ds in enumerate(DATASETS):
            den = col_filt_no_none[ds]
            pct_B_dict[ds] = (filt_no_none_mat[:, i] / den) if den > 0 else np.array([])
    else:
        pct_B_dict = {ds: np.array([]) for ds in DATASETS}

    baseline_label_map = {
        "orig_no_none_pct": "Proportion (A: orig_no_none)",
        "filtered_pct":     "Proportion (B: filtered)"
    }
    plot_title_suffix_map = {
        "orig_no_none_pct": "(orig, excl. <NONE>/Irrelevant)",
        "filtered_pct":     "(filtered topics, excl. <NONE>/Irrelevant)"
    }

    if plt_source == "orig_no_none_pct":
        pct_plot_dict = pct_A_dict
    else:
        pct_plot_dict = pct_B_dict

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Topic"] + DATASETS + ["Total"])

        for tp in sorted(filtered_counts.keys()):
            counts = [filtered_counts[tp].get(ds, 0) for ds in DATASETS]
            row_total = sum(counts)

            if tp in IGNORE_PCT_TOPICS:
                # Counts only, no percentages
                formatted = [f"{cnt}" for cnt in counts]
                writer.writerow([tp, *formatted, f"{row_total}"])
                continue

            # A: orig_no_none_pct, B: filtered_pct
            pct_A = [ (cnt / col_orig_no_none[ds]) if col_orig_no_none[ds] > 0 else 0.0
                      for cnt, ds in zip(counts, DATASETS) ]
            pct_B = [ (cnt / col_filt_no_none[ds])  if col_filt_no_none[ds]  > 0 else 0.0
                      for cnt, ds in zip(counts, DATASETS) ]

            formatted = [
                f"{cnt} ({pA:.2%}; {pB:.2%})"
                for cnt, pA, pB in zip(counts, pct_A, pct_B)
            ]
            total_pct_B = (row_total / global_filt_total_no_ignored) if global_filt_total_no_ignored > 0 else 0.0
            writer.writerow([tp, *formatted, f"{row_total} ({total_pct_B:.2%})"])

        # Variance: based on absolute counts
        var_current  = np.var(filt_no_none_mat, axis=0, ddof=0) if filt_no_none_mat.size else np.zeros(len(DATASETS))
        var_exclNone = np.var(orig_no_none_mat, axis=0, ddof=0) if orig_no_none_mat.size else np.zeros(len(DATASETS))
        var_inclNone = np.var(orig_all_mat,    axis=0, ddof=0) if orig_all_mat.size    else np.zeros(len(DATASETS))

        # Gini: based on absolute counts
        if filt_no_none_mat.size:
            gini_current  = [gini(filt_no_none_mat[:, i]) for i in range(len(DATASETS))]
        else:
            gini_current = [0.0] * len(DATASETS)
        if orig_no_none_mat.size:
            gini_exclNone = [gini(orig_no_none_mat[:, i]) for i in range(len(DATASETS))]
        else:
            gini_exclNone = [0.0] * len(DATASETS)
        if orig_all_mat.size:
            gini_inclNone = [gini(orig_all_mat[:, i]) for i in range(len(DATASETS))]
        else:
            gini_inclNone = [0.0] * len(DATASETS)

        # Percentage matrices (A & B) for summary rows
        def _stack_pct(pct_dict: Dict[str, np.ndarray]) -> np.ndarray:
            vals = [v for v in pct_dict.values() if v.size > 0]
            return np.stack(vals, axis=1) if vals else np.array([])

        pctA_mat = _stack_pct(pct_A_dict)
        pctB_mat = _stack_pct(pct_B_dict)

        if pctA_mat.size:
            varA = np.var(pctA_mat, axis=0, ddof=0)
            gA   = [gini(pctA_mat[:, i]) for i in range(pctA_mat.shape[1])]
        else:
            varA = np.array([])
            gA   = []

        if pctB_mat.size:
            varB = np.var(pctB_mat, axis=0, ddof=0)
            gB   = [gini(pctB_mat[:, i]) for i in range(pctB_mat.shape[1])]
        else:
            varB = np.array([])
            gB   = []

        # Summary rows
        writer.writerow([
            "Var (count)",
            *[f"{v1:.2f};{v2:.2f};{v3:.2f}"
              for v1, v2, v3 in zip(var_current, var_exclNone, var_inclNone)],
            ""
        ])
        writer.writerow([
            "Gini (count)",
            *[f"{g1:.4f};{g2:.4f};{g3:.4f}"
              for g1, g2, g3 in zip(gini_current, gini_exclNone, gini_inclNone)],
            ""
        ])
        writer.writerow([
            "Var (P%)",
            *([f"{a:.6f};{b:.6f}" for a, b in zip(
                varA if varA.size else np.zeros(len(DATASETS)),
                varB if varB.size else np.zeros(len(DATASETS))
            )]),
            ""
        ])
        writer.writerow([
            "Gini (P%)",
            *([f"{a:.4f};{b:.4f}" for a, b in zip(
                gA if gA else [0.0]*len(DATASETS),
                gB if gB else [0.0]*len(DATASETS)
            )]),
            ""
        ])

    print(f"✅ Saved CSV with metrics to {OUT_CSV}")

    # -----------------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------------
    has_pct = any(v.size > 0 for v in pct_plot_dict.values())
    x_label = baseline_label_map.get(plt_source, "Proportion")
    title_suffix = plot_title_suffix_map.get(plt_source, "")

    # KDE
    if has_pct and PLOT_KDE:
        plt.figure()
        for ds, vals in pct_plot_dict.items():
            if vals.size == 0:
                continue
            kde = gaussian_kde(vals)
            xs = np.linspace(0, float(vals.max()), KDE_POINTS)
            ys = kde(xs)
            plt.fill_between(xs, ys, alpha=0.3)
            plt.plot(xs, ys, label=ds)
        plt.xlabel(x_label)
        plt.ylabel("Density")
        plt.title(f"Shaded KDE of Topic Percentage Distributions {title_suffix}")
        plt.legend()
        plt.tight_layout()
        kde_path = OUT_CSV.with_suffix(f".percentage_kde.{plot_format}")
        plt.savefig(kde_path)
        print(f"✅ Saved shaded KDE plot to {kde_path}")

    # Boxplot
    if has_pct and PLOT_BOXPLOT:
        labels, data = zip(*[(ds, vals) for ds, vals in pct_plot_dict.items() if vals.size > 0])
        plt.figure()
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.ylabel(x_label)
        plt.title(f"Boxplot of Topic Proportions by Dataset {title_suffix}")
        plt.tight_layout()
        box_path = OUT_CSV.with_suffix(f".percentage_boxplot.{plot_format}")
        plt.savefig(box_path)
        print(f"✅ Saved boxplot of percentages to {box_path}")

    # Cosine similarity matrix
    if PLOT_SIM_MATRIX:
        cols: List[Tuple[str, np.ndarray]] = [(ds, v) for ds, v in pct_plot_dict.items() if v.size > 0]
        if cols:
            lengths = {len(v) for _, v in cols}
            if len(lengths) == 1:
                S, labels = _cosine_similarity_matrix(cols)
                sim_path = OUT_CSV.with_suffix(f".dataset_similarity.{plot_format}")
                _plot_similarity_matrix(S, labels, sim_path)
            else:
                print("⚠️ Skipped similarity matrix: inconsistent vector lengths.")
        else:
            print("⚠️ Skipped similarity matrix: no percentage data for chosen baseline.")

    if not has_pct:
        print("⚠️ No percentage data available for plotting (all totals are zero).")

if __name__ == "__main__":
    main()
