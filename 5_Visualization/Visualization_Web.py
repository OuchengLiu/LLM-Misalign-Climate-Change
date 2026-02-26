#  python -m streamlit run Visualization_Web.py

import io, re, math
from typing import List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import streamlit as st

# --------------------------- Page ---------------------------
st.set_page_config(
    page_title="Climate QA Corpus Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Palettes (blue/green/orange; no pink) ---------------------------
PASTEL_SERIES = [
    "#9CC9FF", "#7FD1AE", "#FFC781",
    "#7FB3D5", "#A8E6CF", "#FFD39A",
    "#73C6B6", "#AED6F1", "#F9E79F",
    "#82E0AA", "#5DADE2", "#F8C471",
]
def pastel(n:int): return (PASTEL_SERIES * (n // len(PASTEL_SERIES) + 1))[:n]

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blugrn", ["#F7FBFF","#DDEEF7","#BFE3F0","#A3D5E1","#86C7C6","#6BB9A8"]
)
SEQ_CMAP2 = LinearSegmentedColormap.from_list(
    "seq_org", ["#FFF8EC","#FFECC9","#FFE0A8","#FFD38B","#FFC36B","#FFB14A"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_grn_org", ["#2E8B57","#7FD1AE","#F9F9F9","#FFC781","#E67E22"]
)

# --------------------------- Intent/Form mappings ---------------------------
INTENT_MAIN_NAME = {
    "INTENT_1":"Retrieve / Verify Information",
    "INTENT_2":"Analysis / Evaluation",
    "INTENT_3":"Guidance / Support",
    "INTENT_4":"Transformation / Processing",
    "INTENT_5":"Creative / Generative Content",
    "INTENT_6":"Practical / Structured Content",
    "INTENT_7":"Navigation / Access",
    "INTENT_8":"Social / Engagement",
    "INTENT_9":"Others",
}
FORM_MAIN_NAME = {
    "FORM_1":"Concise Direct Answer",
    "FORM_2":"Descriptive / Explanatory Text",
    "FORM_3":"Enumerative / Sequential Structure",
    "FORM_4":"Semi-structured / Structured Data",
    "FORM_5":"Programming Languages",
    "FORM_6":"Markup & Typesetting Languages",
    "FORM_7":"Choice-based",
    "FORM_8":"Multimodal",
    "FORM_9":"Others",
}
INTENT_SUB_BY_MAIN = {
    "INTENT_1":["INTENT_1a","INTENT_1b","INTENT_1c","INTENT_1z"],
    "INTENT_2":["INTENT_2a","INTENT_2b","INTENT_2c"],
    "INTENT_3":["INTENT_3a","INTENT_3b","INTENT_3c","INTENT_3d"],
    "INTENT_4":["INTENT_4a","INTENT_4b","INTENT_4c","INTENT_4e"],
    "INTENT_5":["INTENT_5a","INTENT_5b","INTENT_5c","INTENT_5d","INTENT_5e"],
    "INTENT_6":["INTENT_6a","INTENT_6b","INTENT_6d"],
    "INTENT_7":["INTENT_7a"],
    "INTENT_8":["INTENT_8a","INTENT_8b","INTENT_8c","INTENT_8d"],
    "INTENT_9":["INTENT_9z"],
}
FORM_SUB_BY_MAIN = {
    "FORM_1":["FORM_1a","FORM_1b"],
    "FORM_2":["FORM_2a","FORM_2b"],
    "FORM_3":["FORM_3a","FORM_3b","FORM_3c","FORM_3d","FORM_3e"],
    "FORM_4":["FORM_4a","FORM_4b"],
    "FORM_5":["FORM_5a","FORM_5d","FORM_5z"],
    "FORM_6":["FORM_6a"],
    "FORM_7":["FORM_7a","FORM_7b"],
    "FORM_8":["FORM_8a"],
    "FORM_9":["FORM_9z"],
}

# --------------------------- Utils ---------------------------
def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2 or np.all(arr == 0): return 0.0
    arr = np.sort(arr); n = arr.size; cum = np.cumsum(arr)
    coef = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, min(1.0, coef)))

def safe_listify(x):
    if isinstance(x, list): return x
    if x is None or (isinstance(x, float) and math.isnan(x)): return []
    return [x]

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_.]+", "_", s).strip("_")

# Display helpers for main codes
def intent_main_display(code: str) -> str:
    name = INTENT_MAIN_NAME.get(code, code)
    return f"{code} — {name}"

def form_main_display(code: str) -> str:
    name = FORM_MAIN_NAME.get(code, code)
    return f"{code} — {name}"

# Sorting helpers (category/number order)
def _split_intent_sub(tag:str):
    m = re.match(r"^INTENT_(\d+)([a-z])$", str(tag))
    return (int(m.group(1)), m.group(2)) if m else (999, "z")
def _split_form_sub(tag:str):
    m = re.match(r"^FORM_(\d+)([a-z])$", str(tag))
    return (int(m.group(1)), m.group(2)) if m else (999, "z")
def intent_main_sort_key(tag:str):
    m = re.match(r"^INTENT_(\d+)$", str(tag))
    return int(m.group(1)) if m else 999
def form_main_sort_key(tag:str):
    m = re.match(r"^FORM_(\d+)$", str(tag))
    return int(m.group(1)) if m else 999
def intent_sub_sort_key(tag:str):
    return _split_intent_sub(tag)
def form_sub_sort_key(tag:str):
    return _split_form_sub(tag)
def topic_small_sort_key(label:str):
    m = re.match(r"^\s*([A-F])\s*(\d+)", str(label))
    if m: return (ord(m.group(1)), int(m.group(2)))
    return (ord('Z')+1, 9999)

# --------------------------- Dataset grouping ---------------------------
def infer_dataset_from_id(raw_id: str) -> str:
    if not isinstance(raw_id, str): return "Unknown"
    s = re.sub(r"[\s\-]+", "_", raw_id.lower())
    rules = [
        (r"^wildchat", "WildChat"),
        (r"^lmsys(chat)?", "LMSYSChat"),
        (r"^climateq(&|and)?a|^climate_?qa", "ClimateQ&A"),
        (r"^climaqa[_\-]?gold|climaqa.*gold", "ClimaQA_Gold"),
        (r"^climaqa[_\-]?silver|climaqa.*silver", "ClimaQA_Silver"),
        (r"^climsight|^clim_sight", "ClimSight"),
        (r"^climate[_\-]?fever|^climatefever", "Climate_FEVER"),
        (r"^environmental[_\-]?claims|^env[_\-]?claims", "Environmental_Claims"),
        (r"^scidcc", "SciDCC"),
        (r"^reddit", "Reddit"),
        (r"^ipcc(_| )?ar6|^ipccar6|^ipcc", "IPCC AR6"),
    ]
    for pat, name in rules:
        if re.search(pat, s): return name
    m = re.match(r"^([a-z0-9]+)", s)
    if m: return m.group(1).capitalize()
    return "Unknown"

GROUP_MAP = {
    "WildChat": "Human-to-AI Queries",
    "LMSYSChat": "Human-to-AI Queries",
    "ClimateQ&A": "Human-to-AI Queries",

    "ClimaQA_Gold": "Human-to-AI Guidance Knowledge",
    "ClimaQA_Silver": "Human-to-AI Guidance Knowledge",

    "Reddit": "Human-to-Human Questions",

    "SciDCC": "Human-to-Human Knowledge Provision",
    "IPCC AR6": "Human-to-Human Knowledge Provision",

    "ClimSight": "Auxiliary Corpora",
    "Climate_FEVER": "Auxiliary Corpora",
    "Environmental_Claims": "Auxiliary Corpora",
}

GROUP_ORDER = [
    "Human-to-AI Queries",
    "Human-to-AI Guidance Knowledge",
    "Human-to-Human Questions",
    "Human-to-Human Knowledge Provision",
    "Auxiliary Corpora",
]

DATASET_ORDER_FIXED = list(GROUP_MAP.keys())


def order_datasets_by_group(dnames: List[str]) -> List[str]:
    base = [d for d in DATASET_ORDER_FIXED if d in dnames]

    others = sorted(set(dnames) - set(base))
    return base + others

# --------------------------- Topic parsing ---------------------------
BIG_TOPIC_NAMES = {
    "A": "A. Climate Science Foundations & Methods",
    "B": "B. Ecological Impacts",
    "C": "C. Human Systems & Socioeconomic Impacts",
    "D": "D. Adaptation Strategies",
    "E": "E. Mitigation Mechanisms",
    "F": "F. Others",
}
def parse_topic_label(topic_str: str) -> Tuple[str, str, str]:
    if not isinstance(topic_str, str): return ("F", "F1", "F1. Others")
    m = re.match(r"^\s*([A-F])\s*(\d+)\s*\.?\s*(.*)$", topic_str)
    if m:
        big = m.group(1); num = m.group(2)
        return (big, f"{big}{num}", topic_str.strip())
    return ("F", "F1", "F1. Others")

# --------------------------- Load ---------------------------
@st.cache_data(show_spinner=True)
def load_jsonl(path_or_bytes) -> pd.DataFrame:
    df = pd.read_json(path_or_bytes, lines=True)
    if "dataset" not in df.columns:
        if "id" not in df.columns: raise KeyError("Input JSONL must contain 'id'.")
        df["dataset"] = df["id"].map(infer_dataset_from_id)
    if "dataset_group" not in df.columns:
        df["dataset_group"] = df["dataset"].map(GROUP_MAP).fillna("Unknown")
    if "Final_Topics" in df.columns:
        df["Final_Topics"] = df["Final_Topics"].apply(safe_listify)
    else:
        df["Final_Topics"] = [[] for _ in range(len(df))]
    if "Final_Question_Types" in df.columns:
        def _norm(v):
            out = {"Intent": [], "Form": []}
            if isinstance(v, dict):
                for k in ("Intent","Form"):
                    if k in v and isinstance(v[k], list): out[k]=v[k]
                    elif k in v and pd.notna(v[k]): out[k]=[v[k]]
            return out
        df["Final_Question_Types"] = df["Final_Question_Types"].apply(_norm)
    else:
        df["Final_Question_Types"] = [{"Intent":[],"Form":[]} for _ in range(len(df))]
    return df

# --------------------------- Weighting ---------------------------
WEIGHT_MODES = {
    "Label-count (each label=1; denom=sum of labels)": "label_count",
    "Per-sample (split 1/K to K labels)": "per_sample",
    "Ranked (triangular by order)": "ranked",
}
def compute_weights_for_labels(labels: List[str], mode: str) -> List[float]:
    K = len(labels)
    if K == 0: return []
    if mode == "label_count": return [1.0]*K
    if mode == "per_sample": return [1.0/K]*K
    ranks = list(range(K,0,-1)); s = K*(K+1)/2.0
    return [r/s for r in ranks]

def explode_with_weights(df: pd.DataFrame, list_getter, col_name: str, weight_mode: str):
    rows=[]
    for _, r in df.iterrows():
        labels = list_getter(r)
        weights = compute_weights_for_labels(labels, weight_mode)
        for lab, w in zip(labels, weights):
            rows.append({"id": r["id"], "dataset": r["dataset"], "dataset_group": r["dataset_group"], col_name: lab, "w": w})
    return pd.DataFrame(rows)

# --------------------------- Helpers (drawing & export) ---------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO(); df.to_csv(buf, index=True, encoding="utf-8-sig"); buf.seek(0); return buf.read()

def adaptive_figsize_h(n_items:int, base_w=8, per_item=0.35, min_h=3.0, max_h=8.5):
    h = max(min_h, min(max_h, n_items*per_item))
    return (base_w, h)

# === NEW: single-line tick autosizing with unified x/y font ===
def _autosize_singleline_ticks(fig, ax, rows, cols, xlabels, ylabels,
                               min_size=6, max_size=14, margin_frac=0.92,
                               unify=True):
    """
    Auto-shrink tick fonts so labels fit in a single line without overlapping.
    If unify=True, force x/y to use the same font size (min of the two).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bbox = ax.get_window_extent(renderer=renderer)
    ax_w, ax_h = bbox.width, bbox.height
    cell_w = max(1.0, ax_w / max(1, cols))
    cell_h = max(1.0, ax_h / max(1, rows))

    def _fits(fontsize, which):
        if which == "x":
            max_w = 0
            for lab in ax.get_xticklabels():
                lab.set_fontsize(fontsize)
                bb = lab.get_window_extent(renderer=renderer)
                max_w = max(max_w, bb.width)
            return max_w <= cell_w * margin_frac
        else:
            max_h = 0
            for lab in ax.get_yticklabels():
                lab.set_fontsize(fontsize)
                bb = lab.get_window_extent(renderer=renderer)
                max_h = max(max_h, bb.height)
            return max_h <= cell_h * margin_frac

    fs_x, fs_y = max_size, max_size
    while fs_x > min_size and not _fits(fs_x, "x"):
        fs_x -= 1
    while fs_y > min_size and not _fits(fs_y, "y"):
        fs_y -= 1

    if unify:
        fs = min(fs_x, fs_y)
        fs_x = fs_y = fs

    for lab in ax.get_xticklabels():
        lab.set_fontsize(fs_x)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(fs_y)

    return fs_x, fs_y

def draw_heatmap_square(
    data: pd.DataFrame,
    title: str,
    cmap=SEQ_CMAP,
    cell=0.55,
    show_values=False,
    value_fmt=".2f",
    *,
    max_w=22,
    max_h=26,
    tick_thin_step=1,
    keep_single_line=True,   # single-line ticks
    min_font=6,
    max_font=14,
    margin_frac=0.92,
    **_legacy_kwargs        # swallow legacy args (x_wrap, y_wrap, auto_font, etc.)
):
    rows, cols = data.shape
    figsize = (max(3, min(max_w, cols*cell)), max(3, min(max_h, rows*cell)))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data.values, aspect="equal", cmap=cmap)

    # single-line labels
    xlabels = list(map(str, data.columns))
    ylabels = list(map(str, data.index))

    # optional thinning
    if tick_thin_step > 1:
        def _thin(lbls):
            return [lbl if i % tick_thin_step == 0 else "" for i, lbl in enumerate(lbls)]
        xlabels = _thin(xlabels)
        ylabels = _thin(ylabels)

    ax.set_xticks(np.arange(cols)); ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(rows)); ax.set_yticklabels(ylabels)

    # auto font sizing (single-line, unified)
    _autosize_singleline_ticks(fig, ax, rows, cols, xlabels, ylabels, min_size=min_font, max_size=max_font, margin_frac=margin_frac, unify=True)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_values:
        cur_fs = ax.get_yticklabels()[0].get_fontsize() if ax.get_yticklabels() else 8
        fs_num = max(min_font, cur_fs - 1)
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, format(data.iat[i, j], value_fmt), ha="center", va="center", fontsize=fs_num)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16 if cols>10 else 0.12, left=0.14 if rows>20 else 0.12)
    return fig

def fig_to_bytes(fig, fmt="png")->bytes:
    buf = io.BytesIO(); fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=240); buf.seek(0); return buf.read()

def grouped_barh(ax, labels, A_vals, B_vals, title="", xlabel="Proportion", colors=("C0","C1")):
    n = len(labels); y = np.arange(n); w = 0.38
    ax.barh(y - w/2, A_vals, height=w, alpha=0.95, label="Group A")
    ax.barh(y + w/2, B_vals, height=w, alpha=0.95, label="Group B")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.legend(); ax.grid(axis="x", linestyle="--", alpha=0.25)


# def diverging_bar(ax, labels, values, title="", xlabel="Diff (A - B)"):
#     vals = np.asarray(values, dtype=float)
#     y = np.arange(len(labels))
#     vmax = np.nanmax(np.abs(vals)) if vals.size else 1.0
#     if vmax == 0 or np.isnan(vmax): vmax = 1.0
#     norm = np.clip(vals / vmax, -1, 1)
#     cmap = DIV_CMAP
#     base_colors = cmap((norm + 1) / 2.0)
#     alphas = np.clip(np.abs(vals) / (vmax + 1e-9), 0.35, 1.0)
#     colors = [(c[0], c[1], c[2], float(a)) for c, a in zip(base_colors, alphas)]
#     ax.barh(y, vals, color=colors)
#     ax.axvline(0, color="#555", lw=1)
#     ax.set_yticks(y); ax.set_yticklabels(labels)
#     ax.set_xlabel(xlabel); ax.set_title(title)
#     ax.grid(axis="x", linestyle="--", alpha=0.25)

def diverging_bar(
    ax,
    labels,
    values,
    title="",
    xlabel="Diff (A - B)",
    show_values=False,
    value_fmt=".2f",
    fontsize=8,
):
    vals = np.asarray(values, dtype=float)
    y = np.arange(len(labels))

    vmax = np.nanmax(np.abs(vals)) if vals.size else 1.0
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    norm = np.clip(vals / vmax, -1, 1)
    cmap = DIV_CMAP
    base_colors = cmap((norm + 1) / 2.0)
    alphas = np.clip(np.abs(vals) / (vmax + 1e-9), 0.35, 1.0)
    colors = [(c[0], c[1], c[2], float(a)) for c, a in zip(base_colors, alphas)]

    bars = ax.barh(y, vals, color=colors)

    ax.axvline(0, color="#555", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    if show_values:
        ax.figure.canvas.draw_idle()
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin if xmax > xmin else 1.0
        offset = 0.01 * span

        for bar, v in zip(bars, vals):
            x = bar.get_width()
            y_center = bar.get_y() + bar.get_height() / 2
            ha = "left" if x >= 0 else "right"
            dx = offset if x >= 0 else -offset
            ax.text(
                x + dx,
                y_center,
                format(v, value_fmt),
                va="center",
                ha=ha,
                fontsize=fontsize,
            )


# === Helpers for tables (prop/count matrices) ===
def _others_mask_for_topics(index_labels: List[str]) -> pd.Series:
    # For topics, treat "F" main category as Others.
    return pd.Series([parse_topic_label(s)[0] == "F" for s in index_labels], index=index_labels)

def _others_mask_for_main_codes(index_labels: List[str], prefix: str) -> pd.Series:
    # For Intent main: prefix="INTENT"; Form main: prefix="FORM". Treat *_9 as Others.
    return pd.Series([str(s).startswith(prefix + "_9") for s in index_labels], index=index_labels)

def make_prop_table_from_matrix(count_mat: pd.DataFrame, others_mask: pd.Series) -> pd.DataFrame:
    """
    Build a MultiIndex-column table with per-dataset columns: (dataset, ['prop_all','prop_no_others','count']).
    - prop_all: label / column sum (including Others)
    - prop_no_others: label / column sum after removing rows marked as Others
    - count: raw count/weight in count_mat
    """
    count_mat = count_mat.copy()
    colsum_all = count_mat.sum(axis=0).replace(0, 1.0)
    prop_all = count_mat.div(colsum_all, axis=1)

    mask_keep = ~others_mask.reindex(count_mat.index, fill_value=False)
    count_no_oth = count_mat[mask_keep]
    colsum_nooth = count_no_oth.sum(axis=0).replace(0, 1.0)
    prop_nooth = count_mat.div(colsum_nooth, axis=1)

    ds_order = order_datasets_by_group(list(count_mat.columns))

    arrays0, arrays1 = [], []
    for ds in ds_order:
        arrays0 += [ds, ds, ds]
        arrays1 += ["prop_all", "prop_no_others", "count"]

    out = pd.DataFrame(
        index=count_mat.index,
        columns=pd.MultiIndex.from_arrays([arrays0, arrays1])
    )

    for ds in ds_order:
        out[(ds, "prop_all")] = prop_all.get(ds, np.nan)
        out[(ds, "prop_no_others")] = prop_nooth.get(ds, np.nan)
        out[(ds, "count")] = count_mat.get(ds, np.nan)

    return out

def build_format_dict(df):
    fmt = {}
    for col in df.columns:

        if isinstance(col, tuple):
            if col[1] == "count":
                fmt[col] = "{:.0f}"
            else:
                fmt[col] = "{:.4f}"
        else:
            if col == "count":
                fmt[col] = "{:.0f}"
            else:
                fmt[col] = "{:.4f}"
    return fmt


# --------------------------- Sidebar: Data / Export / Defaults ---------------------------
st.sidebar.header("Data / Export")
default_path = "All_Data_with_Reassigned_Topic_with_QuestionType.jsonl"
choice = st.sidebar.radio("Data source", ["Local path", "Upload JSONL"], index=0, key="sb_data_src")
if choice == "Local path":
    path = st.sidebar.text_input("JSONL file path", value=default_path, key="sb_path")
    if not path.strip(): st.stop()
    try:
        df = load_jsonl(path)
    except Exception as e:
        st.error(f"Failed to load JSONL: {e}"); st.stop()
else:
    up = st.sidebar.file_uploader("Upload JSONL", type=["jsonl"], key="sb_upload")
    if up is None: st.stop()
    try:
        df = load_jsonl(io.BytesIO(up.read()))
    except Exception as e:
        st.error(f"Failed to read uploaded JSONL: {e}"); st.stop()

img_fmt = st.sidebar.selectbox("Image format", ["png","svg"], index=0, key="sb_imgfmt")
st.sidebar.selectbox("Table format", ["csv"], index=0, key="sb_tblfmt")

st.sidebar.header("Global defaults")
default_weight_mode = WEIGHT_MODES[ st.sidebar.selectbox(
    "Default weighting (multi-label)",
    list(WEIGHT_MODES.keys()), index=2, key="sb_weight_mode"
)]
default_cell_size = st.sidebar.slider("Default heatmap cell size (inches)", 0.3, 1.2, 0.55, 0.05, key="sb_cell")
default_show_values = st.sidebar.checkbox("Default: show values on heatmap", value=False, key="sb_showvals")

# === NEW: Global option for excluding 'Others' in proportion computations (for heatmaps/vectors) ===
st.sidebar.header("Proportion Options")
EXCLUDE_OTHERS_FOR_HEATMAP = st.sidebar.checkbox(
    "Exclude 'Others' when computing proportions (heatmaps & vectors)",
    value=False,
    help="Topic: drop F*, Intent: drop INTENT_9*, Form: drop FORM_9* before normalization."
)

# Extractors
def get_topics(row):
    return [t if isinstance(t,str) else "F1. Others" for t in safe_listify(row["Final_Topics"])]
def get_intents(row):
    d = row["Final_Question_Types"]; lst = d.get("Intent", []) if isinstance(d, dict) else []
    return [s if isinstance(s,str) else "INTENT_9z" for s in safe_listify(lst)]
def get_forms(row):
    d = row["Final_Question_Types"]; lst = d.get("Form", []) if isinstance(d, dict) else []
    return [s if isinstance(s,str) else "FORM_9z" for s in safe_listify(lst)]

# Prepare label sets with proper ordering
all_topic_labels = sorted(
    {lab if isinstance(lab,str) else "F1. Others" for rec in df["Final_Topics"] for lab in safe_listify(rec)},
    key=topic_small_sort_key
)
all_intent_sub = sorted(
    {lab for rec in df["Final_Question_Types"] for lab in (rec.get("Intent",[]) if isinstance(rec,dict) else [])},
    key=intent_sub_sort_key
)
all_form_sub = sorted(
    {lab for rec in df["Final_Question_Types"] for lab in (rec.get("Form",[]) if isinstance(rec,dict) else [])},
    key=form_sub_sort_key
)

# --------------------------- Title ---------------------------
st.title("Climate QA Corpus Analysis")
st.caption("Each analysis dimension starts with a comprehensive heatmap; comparisons can be Real vs Bench, Main vs Sub; strict category/index ordering; all metrics are proportions.")

# just for default group seeting, can be changed
REAL_G = "Human-to-AI Queries"
BENCH_G = "Human-to-AI Guidance Knowledge"

# Convenience: dataset order
dataset_order = order_datasets_by_group(sorted(df["dataset"].unique().tolist()))

# ============================================================
#                         TOPIC LEVEL (AD1)
# ============================================================
st.header("Topic Level")

# ---------- Big heatmap before AD1 ----------
st.subheader("Comprehensive Heatmap — Topic (Sub) × Dataset")
with st.expander("Controls • Comprehensive Topic Heatmap", expanded=True):
    col_gt = st.columns(5)
    with col_gt[0]:
        wt_gt = WEIGHT_MODES[st.selectbox("Weighting (topics)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="gt_w")]
    with col_gt[1]:
        topics_sel = st.multiselect("Topics (Sub)", all_topic_labels, default=all_topic_labels, key="gt_topics")
    with col_gt[2]:
        datasets_sel = st.multiselect("Datasets", dataset_order, default=dataset_order, key="gt_datasets")
    with col_gt[3]:
        show_vals_gt = st.checkbox("Show values", value=default_show_values, key="gt_show")
    with col_gt[4]:
        cell_gt = st.slider("Cell size (inches)", 0.3, 1.2, default_cell_size, 0.05, key="gt_cell")

tdf_all = explode_with_weights(df[["id","dataset","dataset_group","Final_Topics"]],
                               get_topics, "topic_label", wt_gt)
tdf_all = tdf_all[(tdf_all["topic_label"].isin(topics_sel)) & (tdf_all["dataset"].isin(datasets_sel))]
ct = tdf_all.groupby(["topic_label","dataset"], as_index=False)["w"].sum()
pv = ct.pivot_table(index="topic_label", columns="dataset", values="w", aggfunc="sum", fill_value=0.0)
pv = pv.reindex(sorted(pv.index, key=topic_small_sort_key))
pv = pv.reindex(columns=order_datasets_by_group(list(pv.columns)))

# === NEW: Table and CSV ===
topic_others_mask = _others_mask_for_topics(list(pv.index))
table_topic = make_prop_table_from_matrix(pv, topic_others_mask)
# st.dataframe(table_topic.style.format({col: "{:.4f}" for col in table_topic.columns if isinstance(col, tuple) and col[1] != "count"}))  # , use_container_width=True)
fmt_topic = build_format_dict(table_topic)
st.dataframe(table_topic.style.format(fmt_topic), use_container_width=True)
st.download_button("Download Topic Table (CSV)", data=df_to_csv_bytes(table_topic), file_name="Table_Topic_vs_Dataset.csv")

# Normalize for heatmap (optionally exclude Others)
pv_for_heat = pv.copy()
if EXCLUDE_OTHERS_FOR_HEATMAP:
    keep_idx = pv_for_heat.index[~topic_others_mask]
    pv_for_heat = pv_for_heat.loc[keep_idx]
for c in pv_for_heat.columns:
    s = pv_for_heat[c].sum(); pv_for_heat[c] = pv_for_heat[c]/(s if s>0 else 1)
fig_gt = draw_heatmap_square(pv_for_heat.T, "Topic (Sub) × Dataset (Proportion)", cmap=SEQ_CMAP, cell=cell_gt, show_values=show_vals_gt, value_fmt=".2f")
st.pyplot(fig_gt)  # , use_container_width=True)
st.download_button("Download Topic Heatmap", data=fig_to_bytes(fig_gt, img_fmt),
                   file_name=f"Heatmap_Topic_Sub_{img_fmt}.{img_fmt}")

# ---------- AD1: Topic distribution — Main vs Sub;----------
st.subheader("AD1: Topic distribution")
with st.expander("Controls • AD1", expanded=True):
    col_ad1_top = st.columns(7)
    with col_ad1_top[0]:
        topic_level = st.radio("Topic level", ["Main (A..F)", "Sub (A1..F?)"], index=1, horizontal=True, key="ad1_level")
    with col_ad1_top[1]:
        chart_type_ad1 = st.selectbox("Chart type", ["Bars + Diff", "Heatmap"], index=0, key="ad1_chart_type")
    with col_ad1_top[2]:
        wt_ad1 = WEIGHT_MODES[st.selectbox("Weighting", list(WEIGHT_MODES.keys()),
                                           index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad1_w")]
    with col_ad1_top[3]:
        dsA = st.multiselect("Group A", dataset_order,
                             default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]), key="ad1_dsA")
    with col_ad1_top[4]:
        dsB = st.multiselect("Group B", dataset_order,
                             default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]), key="ad1_dsB")
    with col_ad1_top[5]:
        show_vals_ad1 = st.checkbox("Show values", value=False, key="ad1_showvals")
    with col_ad1_top[6]:
        cell_ad1 = st.slider("Heatmap cell size", 0.3, 1.2, default_cell_size, 0.05, key="ad1_cell")

tdfA = explode_with_weights(df[df["dataset"].isin(dsA)][["id","dataset","dataset_group","Final_Topics"]],
                            get_topics, "topic_label", wt_ad1)
tdfB = explode_with_weights(df[df["dataset"].isin(dsB)][["id","dataset","dataset_group","Final_Topics"]],
                            get_topics, "topic_label", wt_ad1)

def topic_vector(exploded_df: pd.DataFrame, level: str, sel_topics: List[str]=None) -> pd.Series:
    x = exploded_df.copy()
    if level.startswith("Main"):
        x["topic_main"] = x["topic_label"].apply(lambda s: parse_topic_label(s)[0])
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[x["topic_main"] != "F"]
        s = x.groupby("topic_main")["w"].sum()
        idx = ["A","B","C","D","E","F"]
        for key in idx:
            if key not in s.index: s.loc[key]=0.0
        s = s.reindex(idx)
    else:
        if sel_topics is not None: x = x[x["topic_label"].isin(sel_topics)]
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[x["topic_label"].apply(lambda t: parse_topic_label(t)[0] != "F")]
        s = x.groupby("topic_label")["w"].sum()
        s = s.reindex(sorted(s.index.tolist(), key=topic_small_sort_key))
    tot = s.sum() if s.sum()>0 else 1.0
    return s / tot

if topic_level.startswith("Main"):
    labels = ["A","B","C","D","E","F"]
    A_vec = topic_vector(tdfA, "Main")
    B_vec = topic_vector(tdfB, "Main")
    if chart_type_ad1 == "Bars + Diff":
        fig_ad1_bar, ax_ad1_bar = plt.subplots(figsize=adaptive_figsize_h(len(labels), base_w=8, per_item=0.5))
        grouped_barh(ax_ad1_bar, labels, A_vec.values, B_vec.values, title="Topic Main (A vs B)",
                     xlabel="Proportion")
        if show_vals_ad1:
            y = np.arange(len(labels)); w=0.38
            for i, v in enumerate(A_vec.values): ax_ad1_bar.text(v+0.002, i-w/2, f"{v:.2f}", va="center", fontsize=8)
            for i, v in enumerate(B_vec.values): ax_ad1_bar.text(v+0.002, i+w/2, f"{v:.2f}", va="center", fontsize=8)
        st.pyplot(fig_ad1_bar)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Main Bars", data=fig_to_bytes(fig_ad1_bar, img_fmt),
                           file_name=f"AD1_Topic_Main_Bars.{img_fmt}")

        diff_vals = A_vec - B_vec
        fig_ad1_diff, ax_ad1_diff = plt.subplots(figsize=adaptive_figsize_h(len(labels), base_w=7, per_item=0.45))
        diverging_bar(ax_ad1_diff, labels, diff_vals.values, title="Difference (A - B)", show_values=show_vals_ad1)
        st.pyplot(fig_ad1_diff)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Main Diff", data=fig_to_bytes(fig_ad1_diff, img_fmt),
                           file_name=f"AD1_Topic_Main_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_vec, "Group B":B_vec}, index=labels).T
        fig_ad1_h = draw_heatmap_square(mat, "Topic Main (A vs B) — Heatmap", cmap=SEQ_CMAP, cell=cell_ad1, show_values=show_vals_ad1)
        st.pyplot(fig_ad1_h)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Main Heatmap", data=fig_to_bytes(fig_ad1_h, img_fmt),
                           file_name=f"AD1_Topic_Main_Heatmap.{img_fmt}")
else:
    sel_topics_ad1 = st.multiselect("Topics (Sub) for AD1", all_topic_labels, default=all_topic_labels, key="ad1_topics")
    A_vec = topic_vector(tdfA, "Sub", sel_topics_ad1)
    B_vec = topic_vector(tdfB, "Sub", sel_topics_ad1)
    labels = sorted(set(A_vec.index)|set(B_vec.index), key=topic_small_sort_key)
    A_vals = A_vec.reindex(labels, fill_value=0.0).values
    B_vals = B_vec.reindex(labels, fill_value=0.0).values
    if chart_type_ad1 == "Bars + Diff":
        fig_ad1s_bar, ax_ad1s_bar = plt.subplots(figsize=adaptive_figsize_h(len(labels), base_w=9, per_item=0.35))
        grouped_barh(ax_ad1s_bar, labels, A_vals, B_vals, title="Topic Sub (A vs B)",
                     xlabel="Proportion")
        if show_vals_ad1:
            y = np.arange(len(labels)); w=0.38
            for i, v in enumerate(A_vals): ax_ad1s_bar.text(v+0.002, i-w/2, f"{v:.2f}", va="center", fontsize=7)
            for i, v in enumerate(B_vals): ax_ad1s_bar.text(v+0.002, i+w/2, f"{v:.2f}", va="center", fontsize=7)
        st.pyplot(fig_ad1s_bar)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Sub Bars", data=fig_to_bytes(fig_ad1s_bar, img_fmt),
                           file_name=f"AD1_Topic_Sub_Bars.{img_fmt}")

        diff_vals = (A_vec - B_vec).reindex(labels, fill_value=0.0)
        fig_ad1s_diff, ax_ad1s_diff = plt.subplots(figsize=adaptive_figsize_h(len(labels), base_w=8, per_item=0.32))
        diverging_bar(ax_ad1s_diff, labels, diff_vals.values, title="Difference (A - B)", show_values=show_vals_ad1)
        st.pyplot(fig_ad1s_diff)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Sub Diff", data=fig_to_bytes(fig_ad1s_diff, img_fmt),
                           file_name=f"AD1_Topic_Sub_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_vec.reindex(labels,fill_value=0.0),
                            "Group B":B_vec.reindex(labels,fill_value=0.0)}, index=labels).T
        fig_ad1s_h = draw_heatmap_square(mat, "Topic Sub (A vs B) — Heatmap", cmap=SEQ_CMAP, cell=cell_ad1, show_values=show_vals_ad1)
        st.pyplot(fig_ad1s_h)  # , use_container_width=True)
        st.download_button("Download AD1 Topic Sub Heatmap", data=fig_to_bytes(fig_ad1s_h, img_fmt),
                           file_name=f"AD1_Topic_Sub_Heatmap.{img_fmt}")

# ============================================================
#                    QUESTION TYPE LEVEL (AD3, AD4)
# ============================================================
st.header("Question Type Level")

# ---------- Big heatmaps before AD3 / AD4 ----------
st.subheader("Comprehensive Heatmap — Intent (Sub) × Dataset")
with st.expander("Controls • Comprehensive Intent Heatmap", expanded=True):
    col_gi = st.columns(5)
    with col_gi[0]:
        wt_gi = WEIGHT_MODES[st.selectbox("Weighting (intents)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="gi_w")]
    with col_gi[1]:
        intents_sel = st.multiselect("Intent (Sub)", sorted(set(all_intent_sub), key=intent_sub_sort_key),
                                     default=sorted(set(all_intent_sub), key=intent_sub_sort_key), key="gi_sel")
    with col_gi[2]:
        datasets_sel_gi = st.multiselect("Datasets", dataset_order, default=dataset_order, key="gi_ds")
    with col_gi[3]:
        show_vals_gi = st.checkbox("Show values", value=default_show_values, key="gi_show")
    with col_gi[4]:
        cell_gi = st.slider("Cell size (inches)", 0.3, 1.2, default_cell_size, 0.05, key="gi_cell")

idf_all = explode_with_weights(df[["id","dataset","dataset_group","Final_Question_Types"]],
                               lambda r: [s for s in r["Final_Question_Types"].get("Intent",[])], "Intent", wt_gi)
idf_all["intent_main"] = idf_all["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
idf_all = idf_all[(idf_all["Intent"].isin(intents_sel)) & (idf_all["dataset"].isin(datasets_sel_gi))]
ct_i = idf_all.groupby(["Intent","dataset"], as_index=False)["w"].sum()
pv_i = ct_i.pivot_table(index="Intent", columns="dataset", values="w", aggfunc="sum", fill_value=0.0)
pv_i = pv_i.reindex(sorted(pv_i.index, key=intent_sub_sort_key))
pv_i = pv_i.reindex(columns=order_datasets_by_group(list(pv_i.columns)))

# === NEW: intent Sub table and CSV; identify Others by main code starting with INTENT_9 ===
intent_others_mask = pd.Series([str(x).startswith("INTENT_9") for x in pv_i.index], index=pv_i.index)
table_intent = make_prop_table_from_matrix(pv_i, intent_others_mask)
#st.dataframe(table_intent.style.format({col: "{:.4f}" for col in table_intent.columns if isinstance(col, tuple) and col[1] != "count"}))  # , use_container_width=True)
fmt_intent = build_format_dict(table_intent)
st.dataframe(table_intent.style.format(fmt_intent), use_container_width=True)
st.download_button("Download Intent Table (CSV)", data=df_to_csv_bytes(table_intent), file_name="Table_IntentSub_vs_Dataset.csv")

pv_i_for_heat = pv_i.copy()
if EXCLUDE_OTHERS_FOR_HEATMAP:
    pv_i_for_heat = pv_i_for_heat.loc[~intent_others_mask]
for c in pv_i_for_heat.columns:
    s = pv_i_for_heat[c].sum(); pv_i_for_heat[c] = pv_i_for_heat[c]/(s if s>0 else 1)
fig_gi = draw_heatmap_square(pv_i_for_heat.T, "Intent (Sub) × Dataset (Proportion)", cmap=SEQ_CMAP, cell=cell_gi, show_values=show_vals_gi)
st.pyplot(fig_gi)  # , use_container_width=True)
st.download_button("Download Intent Heatmap", data=fig_to_bytes(fig_gi, img_fmt),
                   file_name=f"Heatmap_Intent_Sub.{img_fmt}")

st.subheader("Comprehensive Heatmap — Form (Sub) × Dataset")
with st.expander("Controls • Comprehensive Form Heatmap", expanded=True):
    col_gf = st.columns(5)
    with col_gf[0]:
        wt_gf = WEIGHT_MODES[st.selectbox("Weighting (forms)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="gf_w")]
    with col_gf[1]:
        forms_sel = st.multiselect("Form (Sub)", sorted(set(all_form_sub), key=form_sub_sort_key),
                                   default=sorted(set(all_form_sub), key=form_sub_sort_key), key="gf_sel")
    with col_gf[2]:
        datasets_sel_gf = st.multiselect("Datasets", dataset_order, default=dataset_order, key="gf_ds")
    with col_gf[3]:
        show_vals_gf = st.checkbox("Show values", value=default_show_values, key="gf_show")
    with col_gf[4]:
        cell_gf = st.slider("Cell size (inches)", 0.3, 1.2, default_cell_size, 0.05, key="gf_cell")

fdf_all = explode_with_weights(df[["id","dataset","dataset_group","Final_Question_Types"]],
                               lambda r: [s for s in r["Final_Question_Types"].get("Form",[])], "Form", wt_gf)
fdf_all["form_main"] = fdf_all["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")
fdf_all = fdf_all[(fdf_all["Form"].isin(forms_sel)) & (fdf_all["dataset"].isin(datasets_sel_gf))]
ct_f = fdf_all.groupby(["Form","dataset"], as_index=False)["w"].sum()
pv_f = ct_f.pivot_table(index="Form", columns="dataset", values="w", aggfunc="sum", fill_value=0.0)
pv_f = pv_f.reindex(sorted(pv_f.index, key=form_sub_sort_key))
pv_f = pv_f.reindex(columns=order_datasets_by_group(list(pv_f.columns)))

# === NEW: form Sub table and CSV; identify Others by main code starting with FORM_9 ===
form_others_mask = pd.Series([str(x).startswith("FORM_9") for x in pv_f.index], index=pv_f.index)
table_form = make_prop_table_from_matrix(pv_f, form_others_mask)
#st.dataframe(table_form.style.format({col: "{:.4f}" for col in table_form.columns if isinstance(col, tuple) and col[1] != "count"}))  # , use_container_width=True)
fmt_form = build_format_dict(table_form)
st.dataframe(table_form.style.format(fmt_form), use_container_width=True)
st.download_button("Download Form Table (CSV)", data=df_to_csv_bytes(table_form), file_name="Table_FormSub_vs_Dataset.csv")

pv_f_for_heat = pv_f.copy()
if EXCLUDE_OTHERS_FOR_HEATMAP:
    pv_f_for_heat = pv_f_for_heat.loc[~form_others_mask]
for c in pv_f_for_heat.columns:
    s = pv_f_for_heat[c].sum(); pv_f_for_heat[c] = pv_f_for_heat[c]/(s if s>0 else 1)
fig_gf = draw_heatmap_square(pv_f_for_heat.T, "Form (Sub) × Dataset (Proportion)", cmap=SEQ_CMAP2, cell=cell_gf, show_values=show_vals_gf)
st.pyplot(fig_gf)  # , use_container_width=True)
st.download_button("Download Form Heatmap", data=fig_to_bytes(fig_gf, img_fmt),
                   file_name=f"Heatmap_Form_Sub.{img_fmt}")

# ---------- AD3: Intent distributions ----------
st.subheader("AD3: Intent distributions")
with st.expander("Controls • AD3", expanded=True):
    col_ad3 = st.columns(7)
    with col_ad3[0]:
        int_level = st.radio("Intent level", ["Main","Sub"], index=0, horizontal=True, key="ad3_lvl")
    with col_ad3[1]:
        chart_type_ad3 = st.selectbox("Chart type", ["Bars + Diff","Heatmap"], 0, key="ad3_chart")
    with col_ad3[2]:
        wt_ad3 = WEIGHT_MODES[st.selectbox("Weighting", list(WEIGHT_MODES.keys()),
                                           index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad3_w")]
    with col_ad3[3]:
        dsA3 = st.multiselect("Group A datasets", dataset_order,
                              default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]), key="ad3_A")
    with col_ad3[4]:
        dsB3 = st.multiselect("Group B datasets", dataset_order,
                              default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]), key="ad3_B")
    with col_ad3[5]:
        show_vals_ad3 = st.checkbox("Show values", value=False, key="ad3_showvals")
    with col_ad3[6]:
        cell_ad3 = st.slider("Heatmap cell size", 0.3, 1.2, default_cell_size, 0.05, key="ad3_cell")

idfA = explode_with_weights(df[df["dataset"].isin(dsA3)][["id","dataset","dataset_group","Final_Question_Types"]],
                            lambda r: [s for s in r["Final_Question_Types"].get("Intent",[])], "Intent", wt_ad3)
idfB = explode_with_weights(df[df["dataset"].isin(dsB3)][["id","dataset","dataset_group","Final_Question_Types"]],
                            lambda r: [s for s in r["Final_Question_Types"].get("Intent",[])], "Intent", wt_ad3)
idfA["intent_main"] = idfA["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
idfB["intent_main"] = idfB["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")

def intent_vector(exploded_df: pd.DataFrame, level: str) -> pd.Series:
    x = exploded_df.copy()
    if level=="Main":
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[x["intent_main"]!="INTENT_9"]
        s = x.groupby("intent_main")["w"].sum()
        idx = [f"INTENT_{i}" for i in range(1,10)]
        for k in idx:
            if k not in s.index: s.loc[k]=0.0
        s = s.reindex(idx)
    else:
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[~x["Intent"].astype(str).str.startswith("INTENT_9")]
        s = x.groupby("Intent")["w"].sum()
        s = s.reindex(sorted(s.index.tolist(), key=intent_sub_sort_key))
    tot = s.sum() if s.sum()>0 else 1.0
    return s / tot

if int_level=="Main":
    labels_i = [f"INTENT_{i}" for i in range(1,10)]
    A_i = intent_vector(idfA, "Main"); B_i = intent_vector(idfB, "Main")
    if chart_type_ad3=="Bars + Diff":
        fig3, ax3 = plt.subplots(figsize=adaptive_figsize_h(len(labels_i), base_w=8, per_item=0.5))
        grouped_barh(ax3, labels_i, A_i.values, B_i.values,
                     title="Intent (Main): A vs B",
                     xlabel="Proportion")

        ax3.set_title("Intent (Main): A vs B")
        if show_vals_ad3:
            y=np.arange(len(labels_i)); w=0.38
            for i, v in enumerate(A_i.values): ax3.text(v+0.003, i-w/2, f"{v:.2f}", va="center", fontsize=8)
            for i, v in enumerate(B_i.values): ax3.text(v+0.003, i+w/2, f"{v:.2f}", va="center", fontsize=8)
        st.pyplot(fig3)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Main Bars", data=fig_to_bytes(fig3, img_fmt),
                           file_name=f"AD3_Intent_Main_Bars.{img_fmt}")

        diff_i = A_i - B_i
        fig3b, ax3b = plt.subplots(figsize=adaptive_figsize_h(len(labels_i), base_w=7, per_item=0.45))
        diverging_bar(ax3b, labels_i, diff_i.values, title="Difference (A - B)", show_values=show_vals_ad3)
        st.pyplot(fig3b)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Main Diff", data=fig_to_bytes(fig3b, img_fmt),
                           file_name=f"AD3_Intent_Main_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_i, "Group B":B_i}, index=labels_i).T
        fig3h = draw_heatmap_square(mat, "Intent (Main) — Heatmap", cmap=SEQ_CMAP, cell=cell_ad3, show_values=show_vals_ad3)
        st.pyplot(fig3h)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Main Heatmap", data=fig_to_bytes(fig3h, img_fmt),
                           file_name=f"AD3_Intent_Main_Heatmap.{img_fmt}")
else:
    A_i = intent_vector(idfA, "Sub"); B_i = intent_vector(idfB, "Sub")
    labels_i = sorted(set(A_i.index)|set(B_i.index), key=intent_sub_sort_key)
    A_vals = A_i.reindex(labels_i, fill_value=0.0).values
    B_vals = B_i.reindex(labels_i, fill_value=0.0).values
    if chart_type_ad3=="Bars + Diff":
        fig3s, ax3s = plt.subplots(figsize=adaptive_figsize_h(len(labels_i), base_w=9, per_item=0.35))
        grouped_barh(ax3s, labels_i, A_vals, B_vals, "Group A","Group B")
        ax3s.set_title("Intent (Sub): A vs B")
        if show_vals_ad3:
            y=np.arange(len(labels_i)); w=0.38
            for i, v in enumerate(A_vals): ax3s.text(v+0.003, i-w/2, f"{v:.2f}", va="center", fontsize=8)
            for i, v in enumerate(B_vals): ax3s.text(v+0.003, i+w/2, f"{v:.2f}", va="center", fontsize=8)

        st.pyplot(fig3s)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Sub Bars", data=fig_to_bytes(fig3s, img_fmt),
                           file_name=f"AD3_Intent_Sub_Bars.{img_fmt}")

        diff_is = (A_i - B_i).reindex(labels_i, fill_value=0.0)
        fig3sd, ax3sd = plt.subplots(figsize=adaptive_figsize_h(len(labels_i), base_w=8, per_item=0.32))
        diverging_bar(ax3sd, labels_i, diff_is.values, title="Difference (A - B)", show_values=show_vals_ad3)
        st.pyplot(fig3sd)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Sub Diff", data=fig_to_bytes(fig3sd, img_fmt),
                           file_name=f"AD3_Intent_Sub_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_i.reindex(labels_i,fill_value=0.0),
                            "Group B":B_i.reindex(labels_i,fill_value=0.0)}, index=labels_i).T
        fig3sh = draw_heatmap_square(mat, "Intent (Sub) — Heatmap", cmap=SEQ_CMAP, cell=cell_ad3, show_values=show_vals_ad3)
        st.pyplot(fig3sh)  # , use_container_width=True)
        st.download_button("Download AD3 Intent Sub Heatmap", data=fig_to_bytes(fig3sh, img_fmt),
                           file_name=f"AD3_Intent_Sub_Heatmap.{img_fmt}")

# ---------- AD4: Form distributions ----------
st.subheader("AD4: Answer Form distributions")
with st.expander("Controls • AD4", expanded=True):
    col_ad4 = st.columns(7)
    with col_ad4[0]:
        form_level = st.radio("Form level", ["Main","Sub"], index=0, horizontal=True, key="ad4_lvl")
    with col_ad4[1]:
        chart_type_ad4 = st.selectbox("Chart type", ["Bars + Diff","Heatmap"], 0, key="ad4_chart")
    with col_ad4[2]:
        wt_ad4 = WEIGHT_MODES[st.selectbox("Weighting", list(WEIGHT_MODES.keys()),
                                           index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad4_w")]
    with col_ad4[3]:
        dsA4 = st.multiselect("Group A datasets", dataset_order,
                              default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]), key="ad4_A")
    with col_ad4[4]:
        dsB4 = st.multiselect("Group B datasets", dataset_order,
                              default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]), key="ad4_B")
    with col_ad4[5]:
        show_vals_ad4 = st.checkbox("Show values", value=False, key="ad4_showvals")
    with col_ad4[6]:
        cell_ad4 = st.slider("Heatmap cell size", 0.3, 1.2, default_cell_size, 0.05, key="ad4_cell")

fdfA = explode_with_weights(df[df["dataset"].isin(dsA4)][["id","dataset","dataset_group","Final_Question_Types"]],
                            lambda r: [s for s in r["Final_Question_Types"].get("Form",[])], "Form", wt_ad4)
fdfB = explode_with_weights(df[df["dataset"].isin(dsB4)][["id","dataset","dataset_group","Final_Question_Types"]],
                            lambda r: [s for s in r["Final_Question_Types"].get("Form",[])], "Form", wt_ad4)
fdfA["form_main"] = fdfA["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")
fdfB["form_main"] = fdfB["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")

def form_vector(exploded_df: pd.DataFrame, level: str) -> pd.Series:
    x = exploded_df.copy()
    if level=="Main":
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[x["form_main"]!="FORM_9"]
        s = x.groupby("form_main")["w"].sum()
        idx = [f"FORM_{i}" for i in range(1,10)]
        for k in idx:
            if k not in s.index: s.loc[k]=0.0
        s = s.reindex(idx)
    else:
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            x = x[~x["Form"].astype(str).str.startswith("FORM_9")]
        s = x.groupby("Form")["w"].sum()
        s = s.reindex(sorted(s.index.tolist(), key=form_sub_sort_key))
    tot = s.sum() if s.sum()>0 else 1.0
    return s / tot

if form_level=="Main":
    labels_f = [f"FORM_{i}" for i in range(1,10)]
    A_f = form_vector(fdfA, "Main"); B_f = form_vector(fdfB, "Main")
    if chart_type_ad4=="Bars + Diff":
        fig4, ax4 = plt.subplots(figsize=adaptive_figsize_h(len(labels_f), base_w=8, per_item=0.5))
        grouped_barh(ax4, labels_f, A_f.values, B_f.values, "Group A","Group B")
        ax4.set_title("Form (Main): A vs B")
        if show_vals_ad4:
            y=np.arange(len(labels_f)); w=0.38
            for i, v in enumerate(A_f.values): ax4.text(v+0.003, i-w/2, f"{v:.2f}", va="center", fontsize=8)
            for i, v in enumerate(B_f.values): ax4.text(v+0.003, i+w/2, f"{v:.2f}", va="center", fontsize=8)
        st.pyplot(fig4)  # , use_container_width=True)
        st.download_button("Download AD4 Form Main Bars", data=fig_to_bytes(fig4, img_fmt),
                           file_name=f"AD4_Form_Main_Bars.{img_fmt}")

        diff_f = A_f - B_f
        fig4b, ax4b = plt.subplots(figsize=adaptive_figsize_h(len(labels_f), base_w=7, per_item=0.45))
        diverging_bar(ax4b, labels_f, diff_f.values, title="Difference (A - B)", show_values=show_vals_ad4)
        st.pyplot(fig4b)  # , use_container_width=True)
        st.download_button("Download AD4 Form Main Diff", data=fig_to_bytes(fig4b, img_fmt),
                           file_name=f"AD4_Form_Main_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_f, "Group B":B_f}, index=labels_f).T
        fig4h = draw_heatmap_square(mat, "Form (Main) — Heatmap", cmap=SEQ_CMAP2, cell=cell_ad4, show_values=show_vals_ad4)
        st.pyplot(fig4h)  # , use_container_width=True)
        st.download_button("Download AD4 Form Main Heatmap", data=fig_to_bytes(fig4h, img_fmt),
                           file_name=f"AD4_Form_Main_Heatmap.{img_fmt}")
else:
    A_f = form_vector(fdfA, "Sub"); B_f = form_vector(fdfB, "Sub")
    labels_f = sorted(set(A_f.index)|set(B_f.index), key=form_sub_sort_key)
    A_vals = A_f.reindex(labels_f, fill_value=0.0).values
    B_vals = B_f.reindex(labels_f, fill_value=0.0).values
    if chart_type_ad4=="Bars + Diff":
        fig4s, ax4s = plt.subplots(figsize=adaptive_figsize_h(len(labels_f), base_w=9, per_item=0.35))
        grouped_barh(ax4s, labels_f, A_vals, B_vals, "Group A","Group B")
        ax4s.set_title("Form (Sub): A vs B")
        if show_vals_ad4:
            y=np.arange(len(labels_f)); w=0.38
            for i, v in enumerate(A_vals): ax4s.text(v+0.003, i-w/2, f"{v:.2f}", va="center", fontsize=8)
            for i, v in enumerate(B_vals): ax4s.text(v+0.003, i+w/2, f"{v:.2f}", va="center", fontsize=8)
        st.pyplot(fig4s)  # , use_container_width=True)
        st.download_button("Download AD4 Form Sub Bars", data=fig_to_bytes(fig4s, img_fmt),
                           file_name=f"AD4_Form_Sub_Bars.{img_fmt}")

        diff_fs = (A_f - B_f).reindex(labels_f, fill_value=0.0)
        fig4sd, ax4sd = plt.subplots(figsize=adaptive_figsize_h(len(labels_f), base_w=8, per_item=0.32))
        diverging_bar(ax4sd, labels_f, diff_fs.values, title="Difference (A - B)", show_values=show_vals_ad4)
        st.pyplot(fig4sd)  # , use_container_width=True)
        st.download_button("Download AD4 Form Sub Diff", data=fig_to_bytes(fig4sd, img_fmt),
                           file_name=f"AD4_Form_Sub_Diff.{img_fmt}")
    else:
        mat = pd.DataFrame({"Group A":A_f.reindex(labels_f,fill_value=0.0),
                            "Group B":B_f.reindex(labels_f,fill_value=0.0)}, index=labels_f).T
        fig4sh = draw_heatmap_square(mat, "Form (Sub) — Heatmap", cmap=SEQ_CMAP2, cell=cell_ad4, show_values=show_vals_ad4)
        st.pyplot(fig4sh)  # , use_container_width=True)
        st.download_button("Download AD4 Form Sub Heatmap", data=fig_to_bytes(fig4sh, img_fmt),
                           file_name=f"AD4_Form_Sub_Heatmap.{img_fmt}")

# ============================================================
#                     CROSS-LEVEL ANALYSIS (AD5–AD7)
# ============================================================
st.header("Cross-level Analysis")

# ---------- Big heatmap before AD5 ----------
st.subheader("Comprehensive Heatmap — Topic (Sub) × Intent (Main)")
with st.expander("Controls • Comprehensive Topic×Intent Heatmap", expanded=True):
    col_ti = st.columns(7)
    with col_ti[0]:
        wt_ti_t = WEIGHT_MODES[st.selectbox("Weight (topics)", list(WEIGHT_MODES.keys()),
                                            index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ti_wt")]
    with col_ti[1]:
        wt_ti_i = WEIGHT_MODES[st.selectbox("Weight (intents)", list(WEIGHT_MODES.keys()),
                                            index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ti_wi")]
    with col_ti[2]:
        topics_ti = st.multiselect("Topics (Sub)", all_topic_labels, default=all_topic_labels, key="ti_topics")
    with col_ti[3]:
        intents_main_all = [f"INTENT_{i}" for i in range(1,10)]
        intents_main_sel = st.multiselect("Intent (Main)", intents_main_all, default=intents_main_all, key="ti_imain")
    with col_ti[4]:
        groups_sel_ti = st.multiselect("Groups (for filter)", GROUP_ORDER[:-1], default=[REAL_G,BENCH_G], key="ti_groups")
    with col_ti[5]:
        show_vals_ti = st.checkbox("Show values", value=default_show_values, key="ti_show")
    with col_ti[6]:
        cell_ti = st.slider("Cell size", 0.3, 1.2, default_cell_size, 0.05, key="ti_cell")

tdf_T = explode_with_weights(df[df["dataset_group"].isin(groups_sel_ti)][["id","dataset","dataset_group","Final_Topics"]],
                             get_topics, "topic_label", wt_ti_t)
idf_I = explode_with_weights(df[df["dataset_group"].isin(groups_sel_ti)][["id","dataset","dataset_group","Final_Question_Types"]],
                             get_intents, "Intent", wt_ti_i)
idf_I["intent_main"] = idf_I["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")

tdf_T = tdf_T[tdf_T["topic_label"].isin(topics_ti)]
idf_I = idf_I[idf_I["intent_main"].isin(intents_main_sel)]

def topic_intent_matrix(expl_tdf: pd.DataFrame, expl_idf: pd.DataFrame) -> pd.DataFrame:
    lt = expl_tdf[["id","topic_label","w"]].rename(columns={"w":"w_t"})
    li = expl_idf[["id","intent_main","w"]].rename(columns={"w":"w_i"})
    merged = lt.merge(li, on="id", how="inner")
    merged["w_joint"] = merged["w_t"] * merged["w_i"]
    ct = merged.groupby(["topic_label","intent_main"], as_index=False)["w_joint"].sum()
    pv = ct.pivot_table(index="topic_label", columns="intent_main", values="w_joint", aggfunc="sum", fill_value=0.0)
    pv = pv.reindex(sorted(pv.index, key=topic_small_sort_key))
    pv = pv.reindex(sorted(pv.columns, key=intent_main_sort_key), axis=1)
    if EXCLUDE_OTHERS_FOR_HEATMAP:
        pv = pv.loc[[parse_topic_label(x)[0]!="F" for x in pv.index], :]
        pv = pv.loc[:, [not str(c).startswith("INTENT_9") for c in pv.columns]]
    pv = pv.div(pv.sum(axis=1).replace(0,1), axis=0)
    return pv

pv_ti_all = topic_intent_matrix(tdf_T, idf_I)
fig_ti_all = draw_heatmap_square(pv_ti_all, "Comprehensive Topic×Intent (Main) — Proportion per Topic row", cmap=SEQ_CMAP, cell=cell_ti, show_values=show_vals_ti)
st.pyplot(fig_ti_all)  # )  # , use_container_width=True)
st.download_button("Download Topic×Intent Heatmap", data=fig_to_bytes(fig_ti_all, img_fmt),
                   file_name=f"Heatmap_Topic_Intent_Main.{img_fmt}")

# ---------- AD5: Topic × Intent — A vs B ----------
st.subheader("AD5: Topic × Intent — Group A vs Group B")
with st.expander("Controls • AD5", expanded=True):
    col5 = st.columns(8)
    with col5[0]:
        tA = st.multiselect("Group A datasets", dataset_order,
                            default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]), key="ad5_tA")
    with col5[1]:
        tB = st.multiselect("Group B datasets", dataset_order,
                            default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]), key="ad5_tB")
    with col5[2]:
        int_lvl = st.radio("Intent level (cols)", ["Main","Sub"], index=0, horizontal=True, key="ad5_ilvl")
    with col5[3]:
        wt_t = WEIGHT_MODES[st.selectbox("Weight (topics)", list(WEIGHT_MODES.keys()),
                                         index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad5_wt")]
    with col5[4]:
        wt_i2 = WEIGHT_MODES[st.selectbox("Weight (intents)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad5_wi")]
    with col5[5]:
        show_vals_ad5 = st.checkbox("Show values", value=False, key="ad5_show")
    with col5[6]:
        cell_ad5 = st.slider("Cell size", 0.3, 1.2, default_cell_size, 0.05, key="ad5_cell")
    with col5[7]:
        sel_topics_ad5 = st.multiselect("Topics (Sub)", all_topic_labels, default=all_topic_labels, key="ad5_topics")

tdf_A = explode_with_weights(df[df["dataset"].isin(tA)][["id","dataset","dataset_group","Final_Topics"]],
                             get_topics, "topic_label", wt_t)
tdf_B = explode_with_weights(df[df["dataset"].isin(tB)][["id","dataset","dataset_group","Final_Topics"]],
                             get_topics, "topic_label", wt_t)
tdf_A = tdf_A[tdf_A["topic_label"].isin(sel_topics_ad5)]
tdf_B = tdf_B[tdf_B["topic_label"].isin(sel_topics_ad5)]
idf_A = explode_with_weights(df[df["dataset"].isin(tA)][["id","dataset","dataset_group","Final_Question_Types"]],
                             get_intents, "Intent", wt_i2)
idf_B = explode_with_weights(df[df["dataset"].isin(tB)][["id","dataset","dataset_group","Final_Question_Types"]],
                             get_intents, "Intent", wt_i2)
idf_A["intent_main"] = idf_A["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
idf_B["intent_main"] = idf_B["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")

def topic_intent_matrix_lvl(tdfX, idfX, level:str):
    lt = tdfX[["id","topic_label","w"]].rename(columns={"w":"w_t"})
    li = idfX[["id","Intent","intent_main","w"]].rename(columns={"w":"w_i"})
    if level=="Main":
        merged = lt.merge(li[["id","intent_main","w_i"]], on="id", how="inner")
        merged["w_joint"] = merged["w_t"] * merged["w_i"]
        ct = merged.groupby(["topic_label","intent_main"], as_index=False)["w_joint"].sum()
        pv = ct.pivot_table(index="topic_label", columns="intent_main", values="w_joint", aggfunc="sum", fill_value=0.0)
        pv = pv.reindex(sorted(pv.index, key=topic_small_sort_key))
        pv = pv.reindex(sorted(pv.columns, key=intent_main_sort_key), axis=1)
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[[parse_topic_label(x)[0]!="F" for x in pv.index], :]
            pv = pv.loc[:, [not str(c).startswith("INTENT_9") for c in pv.columns]]
    else:
        merged = lt.merge(li[["id","Intent","w_i"]], on="id", how="inner")
        merged["w_joint"] = merged["w_t"] * merged["w_i"]
        ct = merged.groupby(["topic_label","Intent"], as_index=False)["w_joint"].sum()
        pv = ct.pivot_table(index="topic_label", columns="Intent", values="w_joint", aggfunc="sum", fill_value=0.0)
        pv = pv.reindex(sorted(pv.index, key=topic_small_sort_key))
        pv = pv.reindex(sorted(pv.columns, key=intent_sub_sort_key), axis=1)
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[[parse_topic_label(x)[0]!="F" for x in pv.index], :]
            pv = pv.loc[:, [not str(c).startswith("INTENT_9") for c in pv.columns]]
    pv = pv.div(pv.sum(axis=1).replace(0,1), axis=0)
    return pv

pv_A = topic_intent_matrix_lvl(tdf_A, idf_A, int_lvl)
pv_B = topic_intent_matrix_lvl(tdf_B, idf_B, int_lvl)

c5 = st.columns(2)
with c5[0]:
    fig5A = draw_heatmap_square(pv_A, f"Topic × Intent ({int_lvl}) — Group A", cmap=SEQ_CMAP, cell=cell_ad5, show_values=show_vals_ad5)
    st.pyplot(fig5A) # )  # , use_container_width=True)
    st.download_button("Download AD5 A Heatmap", data=fig_to_bytes(fig5A, img_fmt),
                       file_name=f"AD5_TopicIntent_{int_lvl}_A.{img_fmt}")
with c5[1]:
    fig5B = draw_heatmap_square(pv_B, f"Topic × Intent ({int_lvl}) — Group B", cmap=SEQ_CMAP, cell=cell_ad5, show_values=show_vals_ad5)
    st.pyplot(fig5B)  # )  # )  # , use_container_width=True)
    st.download_button("Download AD5 B Heatmap", data=fig_to_bytes(fig5B, img_fmt),
                       file_name=f"AD5_TopicIntent_{int_lvl}_B.{img_fmt}")

common_rows = pv_A.index.intersection(pv_B.index); common_cols = pv_A.columns.intersection(pv_B.columns)
diff_mat = pv_A.loc[common_rows, common_cols] - pv_B.loc[common_rows, common_cols]
fig5D = draw_heatmap_square(diff_mat, "Topic × Intent Difference (A - B)", cmap=DIV_CMAP, cell=cell_ad5, show_values=show_vals_ad5)
st.pyplot(fig5D)  # )  # , use_container_width=True)
st.download_button("Download AD5 Diff Heatmap", data=fig_to_bytes(fig5D, img_fmt),
                   file_name=f"AD5_TopicIntent_{int_lvl}_Diff.{img_fmt}")

# ---------- Big heatmap before AD6 ----------
st.subheader("Comprehensive Heatmap — Intent (Main) × Form (Main)")
with st.expander("Controls • Comprehensive Intent×Form Heatmap", expanded=True):
    col_if = st.columns(5)
    with col_if[0]:
        wt_if_i = WEIGHT_MODES[st.selectbox("Weight (intents)", list(WEIGHT_MODES.keys()),
                                            index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="if_wi")]
    with col_if[1]:
        wt_if_f = WEIGHT_MODES[st.selectbox("Weight (forms)", list(WEIGHT_MODES.keys()),
                                            index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="if_wf")]
    with col_if[2]:
        groups_if = st.multiselect("Groups", GROUP_ORDER[:-1], default=[REAL_G,BENCH_G], key="if_groups")
    with col_if[3]:
        show_vals_if = st.checkbox("Show values", value=default_show_values, key="if_show")
    with col_if[4]:
        cell_if = st.slider("Cell size", 0.3, 1.2, default_cell_size, 0.05, key="if_cell")


idf_all2 = explode_with_weights(df[df["dataset_group"].isin(groups_if)][["id","dataset","dataset_group","Final_Question_Types"]],
                                get_intents, "Intent", wt_if_i)
fdf_all2 = explode_with_weights(df[df["dataset_group"].isin(groups_if)][["id","dataset","dataset_group","Final_Question_Types"]],
                                get_forms, "Form", wt_if_f)
idf_all2["intent_main"] = idf_all2["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
fdf_all2["form_main"] = fdf_all2["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")
li = idf_all2[["id","intent_main","w"]].rename(columns={"w":"w_i"})
lf = fdf_all2[["id","form_main","w"]].rename(columns={"w":"w_f"})
merge_if = li.merge(lf, on="id", how="inner")
merge_if["w_joint"] = merge_if["w_i"]*merge_if["w_f"]
ct_if = merge_if.groupby(["intent_main","form_main"], as_index=False)["w_joint"].sum()
pv_if = ct_if.pivot_table(index="intent_main", columns="form_main", values="w_joint", aggfunc="sum", fill_value=0.0)
pv_if = pv_if.reindex(sorted(pv_if.index, key=intent_main_sort_key))
pv_if = pv_if.reindex(sorted(pv_if.columns, key=form_main_sort_key), axis=1)
if EXCLUDE_OTHERS_FOR_HEATMAP:
    pv_if = pv_if.loc[[not str(i).startswith("INTENT_9") for i in pv_if.index], :]
    pv_if = pv_if.loc[:, [not str(c).startswith("FORM_9") for c in pv_if.columns]]
pv_if = pv_if.div(pv_if.sum(axis=1).replace(0,1), axis=0)
fig_if_all = draw_heatmap_square(pv_if, "Comprehensive Intent×Form (Main) — Proportion per Intent row", cmap=SEQ_CMAP2, cell=cell_if, show_values=show_vals_if)
st.pyplot(fig_if_all)  # )  # , use_container_width=True)
st.download_button("Download Intent×Form Heatmap", data=fig_to_bytes(fig_if_all, img_fmt),
                   file_name=f"Heatmap_Intent_Form_Main.{img_fmt}")

# ---------- AD6: Intent × Form — A vs B ----------
st.subheader("AD6: Intent × Form — Group A vs Group B")
with st.expander("Controls • AD6", expanded=True):
    col6 = st.columns(8)
    with col6[0]:
        ifA = st.multiselect("Group A datasets", dataset_order,
                             default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]), key="ad6_ifA")
    with col6[1]:
        ifB = st.multiselect("Group B datasets", dataset_order,
                             default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]), key="ad6_ifB")
    with col6[2]:
        int_lvl6 = st.radio("Intent level (rows)", ["Main","Sub"], index=0, horizontal=True, key="ad6_ilvl")
    with col6[3]:
        form_lvl6 = st.radio("Form level (cols)", ["Main","Sub"], index=0, horizontal=True, key="ad6_flvl")
    with col6[4]:
        wt6_i = WEIGHT_MODES[st.selectbox("Weight (intents)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad6_wi")]
    with col6[5]:
        wt6_f = WEIGHT_MODES[st.selectbox("Weight (forms)", list(WEIGHT_MODES.keys()),
                                          index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad6_wf")]
    with col6[6]:
        show_vals_ad6 = st.checkbox("Show values", value=default_show_values, key="ad6_show")
    with col6[7]:
        cell_ad6 = st.slider("Cell size", 0.3, 1.2, default_cell_size, 0.05, key="ad6_cell")

idf_A2 = explode_with_weights(df[df["dataset"].isin(ifA)][["id","dataset","dataset_group","Final_Question_Types"]],
                              get_intents, "Intent", wt6_i)
idf_B2 = explode_with_weights(df[df["dataset"].isin(ifB)][["id","dataset","dataset_group","Final_Question_Types"]],
                              get_intents, "Intent", wt6_i)
fdf_A2 = explode_with_weights(df[df["dataset"].isin(ifA)][["id","dataset","dataset_group","Final_Question_Types"]],
                              get_forms, "Form", wt6_f)
fdf_B2 = explode_with_weights(df[df["dataset"].isin(ifB)][["id","dataset","dataset_group","Final_Question_Types"]],
                              get_forms, "Form", wt6_f)
idf_A2["intent_main"] = idf_A2["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
idf_B2["intent_main"] = idf_B2["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")
fdf_A2["form_main"] = fdf_A2["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")
fdf_B2["form_main"] = fdf_B2["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")

def intent_form_matrix(idfX, fdfX, int_level, form_level):
    li = idfX[["id","Intent","intent_main","w"]].rename(columns={"w":"w_i"})
    lf = fdfX[["id","Form","form_main","w"]].rename(columns={"w":"w_f"})
    if int_level=="Main":
        left = li[["id","intent_main","w_i"]]; rindex="intent_main"
    else:
        left = li[["id","Intent","w_i"]]; rindex="Intent"
    if form_level=="Main":
        right = lf[["id","form_main","w_f"]]; cindex="form_main"
    else:
        right = lf[["id","Form","w_f"]]; cindex="Form"
    merged = left.merge(right, on="id", how="inner")
    merged["w_joint"] = merged["w_i"] * merged["w_f"]
    ct = merged.groupby([rindex,cindex], as_index=False)["w_joint"].sum()
    pv = ct.pivot_table(index=rindex, columns=cindex, values="w_joint", aggfunc="sum", fill_value=0.0)
    if rindex=="intent_main":
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[[not str(i).startswith("INTENT_9") for i in pv.index], :]
        pv = pv.reindex(sorted(pv.index, key=intent_main_sort_key))
    else:
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[[not str(i).startswith("INTENT_9") for i in pv.index], :]
        pv = pv.reindex(sorted(pv.index, key=intent_sub_sort_key))
    if cindex=="form_main":
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[:, [not str(c).startswith("FORM_9") for c in pv.columns]]
        pv = pv.reindex(sorted(pv.columns, key=form_main_sort_key), axis=1)
    else:
        if EXCLUDE_OTHERS_FOR_HEATMAP:
            pv = pv.loc[:, [not str(c).startswith("FORM_9") for c in pv.columns]]
        pv = pv.reindex(sorted(pv.columns, key=form_sub_sort_key), axis=1)
    pv = pv.div(pv.sum(axis=1).replace(0,1), axis=0)
    return pv

pv_if_A = intent_form_matrix(idf_A2, fdf_A2, int_lvl6, form_lvl6)
pv_if_B = intent_form_matrix(idf_B2, fdf_B2, int_lvl6, form_lvl6)

c6 = st.columns(2)
with c6[0]:
    fig6A = draw_heatmap_square(pv_if_A, f"Intent × Form ({int_lvl6}/{form_lvl6}) — Group A", cmap=SEQ_CMAP, cell=0.42, show_values=show_vals_ad6)
    st.pyplot(fig6A)  #)  # , use_container_width=True)
    st.download_button("Download AD6 A Heatmap", data=fig_to_bytes(fig6A, img_fmt),
                       file_name=f"AD6_IF_{int_lvl6}_{form_lvl6}_A.{img_fmt}")
with c6[1]:
    fig6B = draw_heatmap_square(pv_if_B, f"Intent × Form ({int_lvl6}/{form_lvl6}) — Group B", cmap=SEQ_CMAP2, cell=0.42, show_values=show_vals_ad6)
    st.pyplot(fig6B)  #)  # , use_container_width=True)
    st.download_button("Download AD6 B Heatmap", data=fig_to_bytes(fig6B, img_fmt),
                       file_name=f"AD6_IF_{int_lvl6}_{form_lvl6}_B.{img_fmt}")

common_r = pv_if_A.index.intersection(pv_if_B.index); common_c = pv_if_A.columns.intersection(pv_if_B.columns)
diff_if = pv_if_A.loc[common_r, common_c] - pv_if_B.loc[common_r, common_c]
fig6D = draw_heatmap_square(diff_if, "Intent × Form Difference (A - B)", cmap=DIV_CMAP, cell=0.42, show_values=show_vals_ad6)
st.pyplot(fig6D)  # )  # , use_container_width=True)
st.download_button("Download AD6 Diff Heatmap", data=fig_to_bytes(fig6D, img_fmt),
                   file_name=f"AD6_IF_{int_lvl6}_{form_lvl6}_Diff.{img_fmt}")

# ---------- AD7: Within-Topic Differences — A − B (Group or Dataset; Main or Sub) ----------
st.subheader("AD7: Within-Topic Differences — A − B")

with st.expander("Controls • AD7", expanded=True):
    c = st.columns(9)
    with c[0]:
        topic_for_all = st.selectbox("Pick a topic (Sub)", all_topic_labels, index=0, key="ad7_topic")
    with c[1]:
        compare_mode = st.radio("Compare by", ["Group", "Dataset"], index=0, horizontal=True, key="ad7_mode")
    if compare_mode == "Group":
        group_choices = GROUP_ORDER[:-1]
        with c[2]:
            group_A = st.selectbox("Group A", group_choices,
                                   index=group_choices.index(REAL_G) if REAL_G in group_choices else 0,
                                   key="ad7_groupA")
        with c[3]:
            group_B = st.selectbox("Group B", group_choices,
                                   index=group_choices.index(BENCH_G) if BENCH_G in group_choices else 1,
                                   key="ad7_groupB")
        dsA7, dsB7 = None, None
    else:
        with c[2]:
            dsA7 = st.multiselect("Datasets A", dataset_order,
                                  default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==REAL_G]),
                                  key="ad7_dsA")
        with c[3]:
            dsB7 = st.multiselect("Datasets B", dataset_order,
                                  default=sorted([d for d in dataset_order if GROUP_MAP.get(d)==BENCH_G]),
                                  key="ad7_dsB")
        group_A, group_B = None, None

    with c[4]:
        intent_level_ad7 = st.radio("Intent level", ["Main","Sub"], index=0, horizontal=True, key="ad7_intlvl")
    with c[5]:
        form_level_ad7 = st.radio("Form level", ["Main","Sub"], index=0, horizontal=True, key="ad7_formlvl")
    with c[6]:
        wt_i_ad7 = WEIGHT_MODES[st.selectbox("Weight (intents)", list(WEIGHT_MODES.keys()),
                                             index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad7_wi")]
    with c[7]:
        wt_f_ad7 = WEIGHT_MODES[st.selectbox("Weight (forms)", list(WEIGHT_MODES.keys()),
                                             index=list(WEIGHT_MODES.values()).index(default_weight_mode), key="ad7_wf")]
    with c[8]:
        show_vals_ad7 = st.checkbox("Show values", value=False, key="ad7_showvals")

# show_vals_ad7 = st.checkbox("Show values", value=False, key="ad7_showvals")
tick_thin_ad7 = st.slider("Tick thinning (every N)", 1, 5, 1, 1, key="ad7_thin")

# 1) base scope
if compare_mode == "Group":
    base_df = df[df["dataset_group"].isin([group_A, group_B])].copy()
else:
    base_df = df[df["dataset"].isin((dsA7 or []) + (dsB7 or []))].copy()

# 2) ids in selected topic
tdf_topics_all = explode_with_weights(base_df[["id","dataset","dataset_group","Final_Topics"]],
                                      get_topics, "topic_label", "per_sample")
tdf_topics_all = tdf_topics_all[tdf_topics_all["topic_label"] == topic_for_all][["id","dataset","dataset_group"]].drop_duplicates()

if compare_mode == "Group":
    ids_A = tdf_topics_all[tdf_topics_all["dataset_group"] == group_A][["id"]].assign(bucket="A")
    ids_B = tdf_topics_all[tdf_topics_all["dataset_group"] == group_B][["id"]].assign(bucket="B")
else:
    ids_A = tdf_topics_all[tdf_topics_all["dataset"].isin(dsA7)][["id"]].assign(bucket="A")
    ids_B = tdf_topics_all[tdf_topics_all["dataset"].isin(dsB7)][["id"]].assign(bucket="B")

ids_topic = pd.concat([ids_A, ids_B], axis=0, ignore_index=True).drop_duplicates(subset=["id","bucket"])

# 3) explode intents/forms
idf_all_ad7 = explode_with_weights(base_df[["id","dataset","dataset_group","Final_Question_Types"]],
                                   get_intents, "Intent", wt_i_ad7)
idf_all_ad7["intent_main"] = idf_all_ad7["Intent"].astype(str).str.extract(r'^(INTENT_\d+)', expand=False).fillna("INTENT_9")

fdf_all_ad7 = explode_with_weights(base_df[["id","dataset","dataset_group","Final_Question_Types"]],
                                   get_forms, "Form", wt_f_ad7)
fdf_all_ad7["form_main"] = fdf_all_ad7["Form"].astype(str).str.extract(r'^(FORM_\d+)', expand=False).fillna("FORM_9")

# 4) vector builders
def _within_topic_vector_bucket(ids_df: pd.DataFrame, feat_df: pd.DataFrame, key_col: str, order_func) -> Tuple[pd.Series, pd.Series, List[str]]:
    merged = ids_df.merge(feat_df[["id", key_col, "w"]], on="id", how="inner")
    g = merged.groupby(["bucket", key_col], as_index=False)["w"].sum()
    pv = g.pivot_table(index=key_col, columns="bucket", values="w", aggfunc="sum", fill_value=0.0)
    labels = sorted(pv.index.tolist(), key=order_func)
    pv = pv.reindex(labels)
    A = pv["A"] if "A" in pv.columns else pd.Series(0.0, index=labels)
    B = pv["B"] if "B" in pv.columns else pd.Series(0.0, index=labels)
    # Exclude Others before normalization if switch is on
    if EXCLUDE_OTHERS_FOR_HEATMAP:
        if key_col.startswith("intent"):
            keep = [not str(k).startswith("INTENT_9") for k in labels]
        elif key_col.startswith("form"):
            keep = [not str(k).startswith("FORM_9") for k in labels]
        else:
            keep = [True]*len(labels)
        labels = [lab for lab, k in zip(labels, keep) if k]
        A = A.reindex(labels).fillna(0.0)
        B = B.reindex(labels).fillna(0.0)
    A = A / (A.sum() if A.sum() > 0 else 1.0)
    B = B / (B.sum() if B.sum() > 0 else 1.0)
    return A, B, labels

# Intent vectors (Main/Sub)
if intent_level_ad7 == "Main":
    Ai, Bi, labels_i = _within_topic_vector_bucket(
        ids_topic, idf_all_ad7.rename(columns={"intent_main":"KEY"}).assign(KEY=idf_all_ad7["intent_main"]),
        "KEY", intent_main_sort_key
    )
    display_labels_i = [intent_main_display(x) for x in labels_i]
else:
    Ai, Bi, labels_i = _within_topic_vector_bucket(
        ids_topic, idf_all_ad7.rename(columns={"Intent":"KEY"}).assign(KEY=idf_all_ad7["Intent"]),
        "KEY", intent_sub_sort_key
    )
    display_labels_i = labels_i

diff_i = Ai.reindex(labels_i, fill_value=0.0) - Bi.reindex(labels_i, fill_value=0.0)

# Form vectors (Main/Sub)
if form_level_ad7 == "Main":
    Af, Bf, labels_f = _within_topic_vector_bucket(
        ids_topic, fdf_all_ad7.rename(columns={"form_main":"KEY"}).assign(KEY=fdf_all_ad7["form_main"]),
        "KEY", form_main_sort_key
    )
    display_labels_f = [form_main_display(x) for x in labels_f]
else:
    Af, Bf, labels_f = _within_topic_vector_bucket(
        ids_topic, fdf_all_ad7.rename(columns={"Form":"KEY"}).assign(KEY=fdf_all_ad7["Form"]),
        "KEY", form_sub_sort_key
    )
    display_labels_f = labels_f

diff_f = Af.reindex(labels_f, fill_value=0.0) - Bf.reindex(labels_f, fill_value=0.0)

# 5) plots
label_suffix = (f"{group_A} − {group_B}") if compare_mode=="Group" else "Datasets A − B"
title_topic = f"Within Topic: {topic_for_all}"

c7 = st.columns(2)

with c7[0]:
    if tick_thin_ad7 > 1:
        show_idx = [i for i,_ in enumerate(display_labels_i) if i % tick_thin_ad7 == 0]
        disp_i = [display_labels_i[i] if i in show_idx else "" for i in range(len(display_labels_i))]
    else:
        disp_i = display_labels_i
    fig7i, ax7i = plt.subplots(figsize=adaptive_figsize_h(len(display_labels_i), base_w=8, per_item=0.45))
    diverging_bar(ax7i, disp_i, diff_i.values,
                  title=f"{title_topic} — Intent({intent_level_ad7}) Diff ({label_suffix})",
                  xlabel="Proportion Difference", show_values=show_vals_ad7)
    if show_vals_ad7:
        y = np.arange(len(display_labels_i))
        for i, v in enumerate(diff_i.values):
            ax7i.text(v + (0.01 if v>=0 else -0.01), i, f"{v:.2f}",
                      va="center", ha="left" if v>=0 else "right", fontsize=8)
    st.pyplot(fig7i)  # )  # , use_container_width=True)
    st.download_button(
        "Download AD7 Intent Diff Bars",
        data=fig_to_bytes(fig7i, img_fmt),
        file_name=f"AD7_Intent_Diff_{slugify(topic_for_all)}_{'Group' if compare_mode=='Group' else 'Dataset'}.{img_fmt}"
    )

with c7[1]:
    if tick_thin_ad7 > 1:
        show_idx_f = [i for i,_ in enumerate(display_labels_f) if i % tick_thin_ad7 == 0]
        disp_f = [display_labels_f[i] if i in show_idx_f else "" for i in range(len(display_labels_f))]
    else:
        disp_f = display_labels_f
    fig7f, ax7f = plt.subplots(figsize=adaptive_figsize_h(len(display_labels_f), base_w=8, per_item=0.45))
    diverging_bar(ax7f, disp_f, diff_f.values,
                  title=f"{title_topic} — Form({form_level_ad7}) Diff ({label_suffix})",
                  xlabel="Proportion Difference", show_values=show_vals_ad7)
    if show_vals_ad7:
        y = np.arange(len(display_labels_f))
        for i, v in enumerate(diff_f.values):
            ax7f.text(v + (0.01 if v>=0 else -0.01), i, f"{v:.2f}",
                      va="center", ha="left" if v>=0 else "right", fontsize=8)
    st.pyplot(fig7f)  # )  # , use_container_width=True)
    st.download_button(
        "Download AD7 Form Diff Bars",
        data=fig_to_bytes(fig7f, img_fmt),
        file_name=f"AD7_Form_Diff_{slugify(topic_for_all)}_{'Group' if compare_mode=='Group' else 'Dataset'}.{img_fmt}"
    )

st.success("Done. Sorted by category/index; each AD includes a comprehensive heatmap; flexible comparisons and filters are available.")