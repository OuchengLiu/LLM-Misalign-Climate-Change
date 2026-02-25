import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------
def load_cfg() -> dict:
    """Load Configs.yaml from repo root and return the 'Statistical_Analysis' section."""
    # Adjust 'parents' if your script lives deeper in the tree.
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "Configs.yaml").open("r", encoding="utf-8") as fp:
        root = yaml.safe_load(fp) or {}
    return root.get("4_Type_Modelling", {}).get("Statistical_Analysis", {})

CFG = load_cfg()

# Subject & base folder (optional; useful for defaults)
SUBJECT = CFG.get("subject", "Climate Change")
SUBJECT_SLUG = SUBJECT.replace(" ", "_")

# ---- Paths (all configurable in YAML; sensible defaults provided) ------------
BASE_DIR = Path(
    CFG.get("base_dir", f"../Logs/Test/{SUBJECT_SLUG}")
).expanduser()

DATA_FILE = Path(
    CFG.get("question_type_data_file",
            f"../Logs/Test/{SUBJECT_SLUG}/All_Data_with_Final_Topic_with_QuestionType.jsonl")
).expanduser()

ENRICHED_JSONL = Path(
    CFG.get("question_type_enriched_jsonl",
            f"../Logs/Test/{SUBJECT_SLUG}/All_Data_with_Question_Type_Final.enriched.jsonl")
).expanduser()

INTENT_CSV = Path(
    CFG.get("question_type_intent_csv",
            f"../Logs/Test/{SUBJECT_SLUG}/intent_by_dataset.csv")
).expanduser()

FORM_CSV = Path(
    CFG.get("question_type_form_csv",
            f"../Logs/Test/{SUBJECT_SLUG}/form_by_dataset.csv")
).expanduser()

COMBO_CSV = Path(
    CFG.get("question_type_combo_csv",
            f"../Logs/Test/{SUBJECT_SLUG}/intent_form_combo_by_dataset.csv")
).expanduser()

# ---- Dataset list and matching regex -----------------------------------------
DEFAULT_DATASETS: List[str] = [
    "WildChat", "LMSYSChat", "Reddit", "ClimateQ&A", "ClimSight",
    "ClimaQA_Gold", "ClimaQA_Sliver", "Climate_FEVER",
    "Environmental_Claims", "SciDCC", "IPCC_AR6"
]
DATASETS: List[str] = list(CFG.get("datasets", DEFAULT_DATASETS))

_dataset_rx_str_cfg = CFG.get("dataset_regex", "")
if _dataset_rx_str_cfg:
    DATASET_RX = re.compile(_dataset_rx_str_cfg)
else:
    union = "|".join(re.escape(ds) for ds in DATASETS)
    DATASET_RX = re.compile(rf"^({union})")

# -----------------------------------------------------------------------------
# Label maps (code -> (major_id, major_label, fine_label))
# -----------------------------------------------------------------------------
INTENT_MAP: Dict[str, Tuple[str, str, str]] = {
    # 1 Retrieve / Verify Information
    "INTENT_1a": ("1", "Retrieve / Verify Information", "Fact Lookup"),
    "INTENT_1b": ("1", "Retrieve / Verify Information", "Concept Definition"),
    "INTENT_1c": ("1", "Retrieve / Verify Information", "Clarification / Verification"),
    "INTENT_1z": ("1", "Retrieve / Verify Information", "Others"),
    # 2 Analysis / Evaluation
    "INTENT_2a": ("2", "Analysis / Evaluation", "Reasoning / Causal Analysis"),
    "INTENT_2b": ("2", "Analysis / Evaluation", "Data Analysis / Calculation"),
    "INTENT_2c": ("2", "Analysis / Evaluation", "Evaluation / Review"),
    "INTENT_2z": ("2", "Analysis / Evaluation", "Others"),
    # 3 Guidance / Support
    "INTENT_3a": ("3", "Guidance / Support", "General Advice"),
    "INTENT_3b": ("3", "Guidance / Support", "Technical Assistance / Troubleshooting"),
    "INTENT_3c": ("3", "Guidance / Support", "Planning / Strategy"),
    "INTENT_3d": ("3", "Guidance / Support", "Teaching / Skill Building"),
    "INTENT_3z": ("3", "Guidance / Support", "Others"),
    # 4 Transformation / Processing
    "INTENT_4a": ("4", "Transformation / Processing", "Translation"),
    "INTENT_4b": ("4", "Transformation / Processing", "Rewrite"),
    "INTENT_4c": ("4", "Transformation / Processing", "Summarisation"),
    "INTENT_4d": ("4", "Transformation / Processing", "Format Conversion"),
    "INTENT_4e": ("4", "Transformation / Processing", "Information Extraction"),
    "INTENT_4z": ("4", "Transformation / Processing", "Others"),
    # 5 Creative / Generative Content
    "INTENT_5a": ("5", "Creative / Generative Content", "General Text"),
    "INTENT_5b": ("5", "Creative / Generative Content", "Creative Story / Poem / Lyrics"),
    "INTENT_5c": ("5", "Creative / Generative Content", "Hypothetical Scenario"),
    "INTENT_5d": ("5", "Creative / Generative Content", "Role-play / Dialogue Simulation"),
    "INTENT_5e": ("5", "Creative / Generative Content", "Multimodal Content Creation"),
    "INTENT_5z": ("5", "Creative / Generative Content", "Others"),
    # 6 Practical / Structured Content
    "INTENT_6a": ("6", "Practical / Structured Content", "Operational Writing"),
    "INTENT_6b": ("6", "Practical / Structured Content", "Code Solution"),
    "INTENT_6c": ("6", "Practical / Structured Content", "Formula & Expressions"),
    "INTENT_6d": ("6", "Practical / Structured Content", "Structured Generation"),
    "INTENT_6z": ("6", "Practical / Structured Content", "Others"),
    # 7 Navigation / Access
    "INTENT_7a": ("7", "Navigation / Access", "Website Navigation"),
    "INTENT_7b": ("7", "Navigation / Access", "System / Resource Access"),
    "INTENT_7z": ("7", "Navigation / Access", "Others"),
    # 8 Social / Engagement
    "INTENT_8a": ("8", "Social / Engagement", "Greeting / Small Talk"),
    "INTENT_8b": ("8", "Social / Engagement", "Entertainment / Engagement"),
    "INTENT_8c": ("8", "Social / Engagement", "Emotional Support / Empathy"),
    "INTENT_8z": ("8", "Social / Engagement", "Others"),
    # 9 Others
    "INTENT_9z": ("9", "Others", "Others"),
}

FORM_MAP: Dict[str, Tuple[str, str, str]] = {
    # 1 Concise Direct Answer
    "FORM_1a": ("1", "Concise Direct Answer", "Concise Value(s) / Entity(ies)"),
    "FORM_1b": ("1", "Concise Direct Answer", "Brief Statement"),
    "FORM_1z": ("1", "Concise Direct Answer", "Others"),
    # 2 Descriptive / Explanatory Text
    "FORM_2a": ("2", "Descriptive / Explanatory Text", "Concise Paragraph"),
    "FORM_2b": ("2", "Descriptive / Explanatory Text", "Detailed Multi-paragraph"),
    "FORM_2z": ("2", "Descriptive / Explanatory Text", "Others"),
    # 3 Enumerative / Sequential Structure
    "FORM_3a": ("3", "Enumerative / Sequential Structure", "Item List"),
    "FORM_3b": ("3", "Enumerative / Sequential Structure", "Comparison List"),
    "FORM_3c": ("3", "Enumerative / Sequential Structure", "Timeline / Flow"),
    "FORM_3d": ("3", "Enumerative / Sequential Structure", "Procedural Steps"),
    "FORM_3e": ("3", "Enumerative / Sequential Structure", "Plan / Itinerary"),
    "FORM_3z": ("3", "Enumerative / Sequential Structure", "Others"),
    # 4 Semi-structured / Structured Data
    "FORM_4a": ("4", "Semi-structured / Structured Data", "Tabular Data"),
    "FORM_4b": ("4", "Semi-structured / Structured Data", "JSON"),
    "FORM_4c": ("4", "Semi-structured / Structured Data", "XML / YAML"),
    "FORM_4z": ("4", "Semi-structured / Structured Data", "Others"),
    # 5 Programming Languages
    "FORM_5a": ("5", "Programming Languages", "General-purpose Languages"),
    "FORM_5b": ("5", "Programming Languages", "Data & Query / Text-matching DSLs"),
    "FORM_5c": ("5", "Programming Languages", "Build & Automation DSLs"),
    "FORM_5d": ("5", "Programming Languages", "Configuration-as-Code / Infra DSLs"),
    "FORM_5e": ("5", "Programming Languages", "Hardware Description Languages"),
    "FORM_5z": ("5", "Programming Languages", "Others"),
    # 6 Markup & Typesetting Languages
    "FORM_6a": ("6", "Markup & Typesetting Languages", "Semantic Document Markup"),
    "FORM_6b": ("6", "Markup & Typesetting Languages", "Styling & Layout"),
    "FORM_6c": ("6", "Markup & Typesetting Languages", "Math & Formula Notation"),
    "FORM_6d": ("6", "Markup & Typesetting Languages", "Diagrammatic Markup"),
    "FORM_6e": ("6", "Markup & Typesetting Languages", "Metadata & Citations"),
    "FORM_6z": ("6", "Markup & Typesetting Languages", "Others"),
    # 7 Choice-based
    "FORM_7a": ("7", "Choice-based", "Multiple Choice"),
    "FORM_7b": ("7", "Choice-based", "Yes/No / True/False"),
    "FORM_7c": ("7", "Choice-based", "Ranking / Ordering"),
    "FORM_7d": ("7", "Choice-based", "Matching"),
    "FORM_7z": ("7", "Choice-based", "Others"),
    # 8 Multimodal
    "FORM_8a": ("8", "Multimodal", "Image"),
    "FORM_8b": ("8", "Multimodal", "Audio"),
    "FORM_8c": ("8", "Multimodal", "Video"),
    "FORM_8z": ("8", "Multimodal", "Others"),
    # 9 Others
    "FORM_9z": ("9", "Others", "Others"),
}

OUTPUT_DIR = (BASE_DIR / "QuestionType_Analysis")
FIG_DIR = OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Allow choosing a nicer colormap for similarity heatmaps via YAML:
# e.g., in Configs.yaml -> 4_Type_Modelling -> Statistical_Analysis:
#   similarity_cmap: "plasma"  (or "magma", "cividis", etc.)
#   similarity_cmap: "YlOrBr"  (or "Oranges", "YlOrRd", etc.)
SIM_CMAP = str(CFG.get("similarity_cmap", "Oranges"))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def dataset_from_id(sample_id: str) -> str:
    """
    Infer dataset name by regex configured in YAML.
    If no regex match, fallback to token before first underscore.
    """
    m = DATASET_RX.search(sample_id or "")
    if m:
        return m.group(1)
    return sample_id.split("_")[0] if "_" in sample_id else sample_id

def pct(n: int, d: int) -> float:
    """Percentage helper (0 if denominator is 0)."""
    return (n / d * 100.0) if d else 0.0

def ensure_all_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all configured dataset columns exist and are in the configured order."""
    for ds in DATASETS:
        if ds not in df.columns:
            df[ds] = 0
    return df[DATASETS]

# ------------------------- NEW UTILS FOR MULTI-LABEL --------------------------
def as_code_list(x):
    """
    Convert input to a clean list of codes (strip spaces, deduplicate, keep order).
    Accepts list/str/None.
    """
    out = []
    if x is None:
        return out
    if isinstance(x, list):
        it = x
    elif isinstance(x, str):
        it = [x]
    else:
        return out
    seen = set()
    for v in it:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def map_intent_label(code: str) -> Tuple[str, str, str, str]:
    """
    Map an intent code to (major_id, major_label, fine_label, display_label).
    If code is not in INTENT_MAP, mark as '(UNMAPPED)' for visibility.
    """
    major_id, major_label, fine_label = INTENT_MAP.get(code, ("?", "Unknown", "Unknown"))
    if code in INTENT_MAP:
        display = f"{code}. {fine_label}"
    else:
        display = f"{code}. (UNMAPPED)"
    return major_id, major_label, fine_label, display

def map_form_label(code: str) -> Tuple[str, str, str, str]:
    """
    Map a form code to (major_id, major_label, fine_label, display_label).
    If code is not in FORM_MAP, mark as '(UNMAPPED)' for visibility.
    """
    major_id, major_label, fine_label = FORM_MAP.get(code, ("?", "Unknown", "Unknown"))
    if code in FORM_MAP:
        display = f"{code}. {fine_label}"
    else:
        display = f"{code}. (UNMAPPED)"
    return major_id, major_label, fine_label, display

# -----------------------------------------------------------------------------
# (kept for compatibility, but NOT used in multi-label mode)
def build_count_pct_table(
    df: pd.DataFrame,
    label_col: str,     # fine-grained label column for row index (e.g., "intent_label")
    major_col: str,     # major-category column for summary (e.g., "intent_major")
) -> pd.DataFrame:
    """
    Original single-label helper (kept for compatibility).
    Not used in the multi-label pipeline below.
    """
    counts = df.pivot_table(index=label_col, columns="dataset", values="id", aggfunc="count", fill_value=0)
    counts = ensure_all_dataset_columns(counts)
    totals = counts.sum(axis=0)

    pct_df = counts.copy()
    for ds in DATASETS:
        denom = totals.get(ds, 0)
        pct_df[ds] = counts[ds].apply(lambda x: pct(x, denom))

    assembled = pd.DataFrame(index=counts.index)
    for ds in DATASETS:
        assembled[f"{ds} (count)"] = counts[ds]
        assembled[f"{ds} (%)"] = pct_df[ds]

    major_counts = df.pivot_table(index=major_col, columns="dataset", values="id", aggfunc="count", fill_value=0)
    major_counts = ensure_all_dataset_columns(major_counts)

    major_pct = major_counts.copy()
    for ds in DATASETS:
        denom = totals.get(ds, 0)
        major_pct[ds] = major_counts[ds].apply(lambda x: pct(x, denom))

    major_assembled = pd.DataFrame(index=major_counts.index)
    for ds in DATASETS:
        major_assembled[f"{ds} (count)"] = major_counts[ds]
        major_assembled[f"{ds} (%)"] = major_pct[ds]

    out_df = pd.concat([assembled, pd.DataFrame({"": []}), major_assembled], axis=0)
    return out_df

def _ensure_prop_table(df: pd.DataFrame, label_col: str, datasets: List[str]) -> pd.DataFrame:
    """
    Build a proportion table (rows = categories, columns = datasets) from an
    occurrence dataframe. Denominator per dataset is the total #occurrences.
    This is multi-label friendly; sums may exceed 1 across categories.
    """
    if df.empty:
        return pd.DataFrame()

    counts = df.pivot_table(index=label_col, columns="dataset", values="id",
                            aggfunc="count", fill_value=0)
    # Ensure all dataset columns exist and keep the configured order
    for ds in datasets:
        if ds not in counts.columns:
            counts[ds] = 0
    counts = counts[datasets]

    totals = counts.sum(axis=0)  # denominators per dataset
    props = counts.divide(totals.replace(0, np.nan), axis=1).fillna(0.0)
    return props

def _collapse_top_n(props: pd.DataFrame, top_n: int = None, other_label: str = "Other") -> pd.DataFrame:
    """
    Optionally keep only the top-N categories (by mean proportion across datasets)
    and merge the rest into a single 'Other' row to avoid over-crowded plots.
    """
    if props.empty or not top_n or top_n <= 0 or top_n >= len(props):
        return props

    order = props.mean(axis=1).sort_values(ascending=False)
    keep = order.index[:top_n]
    drop = order.index[top_n:]

    kept = props.loc[keep]
    if len(drop) > 0:
        other = props.loc[drop].sum(axis=0).to_frame().T
        other.index = [other_label]
        return pd.concat([kept, other], axis=0)
    return kept

def plot_stacked_bars(props: pd.DataFrame,
                      title: str,
                      file_basename: str,
                      rotation: int = 35,
                      cmap_name: str = "tab20"):
    """
    Plot stacked bars: each dataset is one bar; stacks are category proportions.

    Features:
    - X tick labels slanted, bars spaced apart.
    - Stack order: label name ascending (smallest label top, largest bottom).
    - Unique colors for each label, drawn from a colormap (default 'tab20').
    """
    if props.empty:
        print(f"[plot] Skip {title}: empty props")
        return

    # Order categories by label name (lexicographic ascending)
    sorted_rows = sorted(list(props.index), key=lambda s: str(s).lower())
    draw_order = list(reversed(sorted_rows))  # reverse so smallest on top
    props = props.loc[draw_order]

    # Generate unique colors for all categories
    cmap = cm.get_cmap(cmap_name, len(props.index))
    colors = {cat: cmap(i) for i, cat in enumerate(props.index)}

    # Plot
    fig_w = max(6, len(props.columns) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bottoms = np.zeros(len(props.columns))
    x = np.arange(len(props.columns))
    bar_width = 0.78
    ax.margins(x=0.06)

    for cat, row in props.iterrows():
        ax.bar(x, row.values, bottom=bottoms, width=bar_width,
               label=str(cat), color=colors[cat])
        bottoms += row.values

    ax.set_xticks(x)
    ax.set_xticklabels(props.columns, rotation=rotation, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title(title)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    fig.tight_layout()

    out_png = FIG_DIR / f"{file_basename}.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[plot] Saved: {out_png.resolve()}")


def cosine_similarity_matrix(props: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between datasets based on proportion vectors.
    """
    if props.empty:
        return pd.DataFrame()
    X = props.fillna(0.0).to_numpy().T  # shape: [num_datasets, num_categories]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    return pd.DataFrame(sim, index=props.columns, columns=props.columns)

def plot_similarity_heatmap(sim: pd.DataFrame, title: str, file_basename: str):
    """
    Plot a heatmap for similarity matrix using a pleasant colormap (default 'viridis').
    """
    if sim.empty:
        print(f"[plot] Skip {title}: empty similarity")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(sim) * 0.6), max(5, len(sim) * 0.5)))
    im = ax.imshow(sim.values, aspect="auto", cmap=SIM_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(sim.columns)))
    ax.set_yticks(np.arange(len(sim.index)))
    ax.set_xticklabels(sim.columns, rotation=45, ha="right")
    ax.set_yticklabels(sim.index)
    ax.set_title(title)

    # Annotate values for easier reading (optional, can be toggled via YAML later)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity (0–1)")

    fig.tight_layout()
    out_png = FIG_DIR / f"{file_basename}.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[plot] Saved: {out_png.resolve()}")

def build_props_for_intent(intent_df: pd.DataFrame,
                           level: str,  # "major" or "fine"
                           datasets: List[str],
                           top_n: int = None) -> pd.DataFrame:
    """
    Build intent proportion table at the desired granularity.
    - level='major' -> uses 'intent_major'
    - level='fine'  -> uses 'intent_label'
    """
    if intent_df.empty:
        return pd.DataFrame()
    col = "intent_major" if level == "major" else "intent_label"
    props = _ensure_prop_table(intent_df, col, datasets)
    if level == "fine":
        props = _collapse_top_n(props, top_n=top_n, other_label="Other (fine)")
    return props

def build_props_for_form(form_df: pd.DataFrame,
                         level: str,  # "major" or "fine"
                         datasets: List[str],
                         top_n: int = None) -> pd.DataFrame:
    """
    Build form proportion table at the desired granularity.
    - level='major' -> uses 'form_major'
    - level='fine'  -> uses 'form_label'
    """
    if form_df.empty:
        return pd.DataFrame()
    col = "form_major" if level == "major" else "form_label"
    props = _ensure_prop_table(form_df, col, datasets)
    if level == "fine":
        props = _collapse_top_n(props, top_n=top_n, other_label="Other (fine)")
    return props


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # Ensure parent folders for outputs exist
    for p in [ENRICHED_JSONL, INTENT_CSV, FORM_CSV, COMBO_CSV]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # MULTI-LABEL PIPELINE:
    # Build three occurrence tables (one row per occurrence), not one row per sample.
    #   1) intent_occurs: each row = one intent assigned in a sample
    #   2) form_occurs:   each row = one form   assigned in a sample
    #   3) combo_occurs:  each row = one (intent, form) pair in a sample (Cartesian product)
    # Also write an enriched JSONL where Question_Type_Final lists all form display labels (joined by '; ').
    # -----------------------------------------------------------------------------
    intent_rows = []
    form_rows = []
    combo_rows = []

    with DATA_FILE.open("r", encoding="utf-8") as fp, ENRICHED_JSONL.open("w", encoding="utf-8") as fout:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines; switch to strict mode if needed.
                continue

            sample_id = obj.get("id", "")
            ds = dataset_from_id(sample_id)

            # Expect list/str/None; normalize to clean list of codes
            raw_intents = as_code_list(obj.get("Question Type", {}).get("Intent"))
            raw_forms   = as_code_list(obj.get("Question Type", {}).get("Form"))

            # ---- Intent occurrences
            for ic in raw_intents:
                major_id, major_label, _, disp = map_intent_label(ic)
                intent_rows.append({
                    "id": sample_id,
                    "dataset": ds,
                    "intent_code": ic,
                    "intent_label": disp,                 # e.g., "INTENT_2a Reasoning..." or "INTENT_xx (UNMAPPED)"
                    "intent_major": f"INTENT_{major_id}* {major_label}",
                })

            # ---- Form occurrences
            for fc in raw_forms:
                major_id, major_label, _, disp = map_form_label(fc)
                form_rows.append({
                    "id": sample_id,
                    "dataset": ds,
                    "form_code": fc,
                    "form_label": disp,                   # e.g., "FORM_7a Multiple Choice" or "FORM_xx (UNMAPPED)"
                    "form_major": f"FORM_{major_id}* {major_label}",
                })

            # ---- Combo occurrences: Cartesian product of (intents x forms)
            if raw_intents and raw_forms:
                for ic in raw_intents:
                    imaj_id, imaj_label, _, ilabel = map_intent_label(ic)
                    for fc in raw_forms:
                        fmaj_id, fmaj_label, _, flabel = map_form_label(fc)
                        combo_rows.append({
                            "id": sample_id,
                            "dataset": ds,
                            "intent_code": ic,
                            "intent_label": ilabel,
                            "intent_major": f"INTENT_{imaj_id}* {imaj_label}",
                            "form_code": fc,
                            "form_label": flabel,
                            "form_major": f"FORM_{fmaj_id}* {fmaj_label}",
                            "combo_label": f"{ilabel} + {flabel}",
                            "combo_major": f"INTENT_{imaj_id}* {imaj_label} + FORM_{fmaj_id}* {fmaj_label}",
                        })

            # ---- Enriched JSONL: write all form labels (joined by '; ') for downstream usage
            form_labels_for_json = []
            for fc in raw_forms:
                _, _, _, flabel = map_form_label(fc)
                form_labels_for_json.append(flabel)
            obj["Question_Type_Final"] = "; ".join(form_labels_for_json) if form_labels_for_json else "UNKNOWN"
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Convert to DataFrames (allow empty)
    intent_df = pd.DataFrame(intent_rows) if intent_rows else pd.DataFrame(columns=["id","dataset","intent_label","intent_major"])
    form_df   = pd.DataFrame(form_rows)   if form_rows   else pd.DataFrame(columns=["id","dataset","form_label","form_major"])
    combo_df  = pd.DataFrame(combo_rows)  if combo_rows  else pd.DataFrame(columns=["id","dataset","combo_label","combo_major"])

    # -----------------------------------------------------------------------------
    # Build "count + %" tables for each occurrence table.
    # Denominator for % is the total number of occurrences (per dataset) in that table.
    # -----------------------------------------------------------------------------
    def count_pct(df: pd.DataFrame, label_col: str, major_col: str, datasets: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        # Fine-level counts
        counts = df.pivot_table(index=label_col, columns="dataset", values="id", aggfunc="count", fill_value=0)
        counts = ensure_all_dataset_columns(counts)
        totals = counts.sum(axis=0)  # denominators (per dataset)

        pct_df = counts.copy()
        for ds in datasets:
            denom = totals.get(ds, 0)
            pct_df[ds] = counts[ds].apply(lambda x: pct(x, denom))

        # Interleave columns: "<Dataset> (count)" and "<Dataset> (%)"
        assembled = pd.DataFrame(index=counts.index)
        for ds in datasets:
            assembled[f"{ds} (count)"] = counts[ds]
            assembled[f"{ds} (%)"] = pct_df[ds]

        # Major-level summary with the same denominators
        major_counts = df.pivot_table(index=major_col, columns="dataset", values="id", aggfunc="count", fill_value=0)
        major_counts = ensure_all_dataset_columns(major_counts)

        major_pct = major_counts.copy()
        for ds in datasets:
            denom = totals.get(ds, 0)
            major_pct[ds] = major_counts[ds].apply(lambda x: pct(x, denom))

        major_assembled = pd.DataFrame(index=major_counts.index)
        for ds in datasets:
            major_assembled[f"{ds} (count)"] = major_counts[ds]
            major_assembled[f"{ds} (%)"] = major_pct[ds]

        out_df = pd.concat([assembled, pd.DataFrame({"": []}), major_assembled], axis=0)
        return out_df

    # Intent and form outputs
    intent_out_df = count_pct(intent_df, "intent_label", "intent_major", DATASETS)
    form_out_df   = count_pct(form_df,   "form_label",   "form_major",   DATASETS)

    # Combo output: use (intent, form) occurrences; denominator = total # of combos per dataset
    if not combo_df.empty:
        combo_counts = combo_df.pivot_table(index="combo_label", columns="dataset", values="id", aggfunc="count", fill_value=0)
        combo_counts = ensure_all_dataset_columns(combo_counts)
        combo_totals = combo_counts.sum(axis=0)

        combo_pct = combo_counts.copy()
        for ds in DATASETS:
            denom = combo_totals.get(ds, 0)
            combo_pct[ds] = combo_counts[ds].apply(lambda x: pct(x, denom))

        combo_assembled = pd.DataFrame(index=combo_counts.index)
        for ds in DATASETS:
            combo_assembled[f"{ds} (count)"] = combo_counts[ds]
            combo_assembled[f"{ds} (%)"] = combo_pct[ds]

        combo_major_counts = combo_df.pivot_table(index="combo_major", columns="dataset", values="id", aggfunc="count", fill_value=0)
        combo_major_counts = ensure_all_dataset_columns(combo_major_counts)
        combo_major_pct = combo_major_counts.copy()
        for ds in DATASETS:
            denom = combo_totals.get(ds, 0)
            combo_major_pct[ds] = combo_major_counts[ds].apply(lambda x: pct(x, denom))

        combo_major_assembled = pd.DataFrame(index=combo_major_counts.index)
        for ds in DATASETS:
            combo_major_assembled[f"{ds} (count)"] = combo_major_counts[ds]
            combo_major_assembled[f"{ds} (%)"] = combo_major_pct[ds]

        combo_out_df = pd.concat([combo_assembled, pd.DataFrame({"": []}), combo_major_assembled], axis=0)
    else:
        # No (intent, form) pairs exist in the data
        combo_out_df = pd.DataFrame()

    # Write CSVs (write even if empty, to keep pipeline stable)
    intent_out_df.to_csv(INTENT_CSV, index=True, encoding="utf-8-sig")
    form_out_df.to_csv(FORM_CSV, index=True, encoding="utf-8-sig")
    if combo_out_df is not None:
        combo_out_df.to_csv(COMBO_CSV, index=True, encoding="utf-8-sig")

    # Print locations for convenience
    print("Done.")
    print(f"Enriched JSONL: {ENRICHED_JSONL.resolve()}")
    print(f"Intent CSV:     {INTENT_CSV.resolve()}")
    print(f"Form CSV:       {FORM_CSV.resolve()}")
    print(f"Combo CSV:      {COMBO_CSV.resolve()}")

    # ----------------------------- PLOTS & MATRICES ----------------------------
    # Fine-level can be very crowded; keep top-N and merge the rest to "Other".
    TOP_N_FINE = int(CFG.get("plot_top_n_fine", 15))

    # Intent: major / fine stacked bars
    intent_props_major = build_props_for_intent(intent_df, level="major", datasets=DATASETS)
    plot_stacked_bars(intent_props_major,
                      title="Intent (Major) Proportions by Dataset",
                      file_basename="intent_major_stacked")

    intent_props_fine = build_props_for_intent(intent_df, level="fine", datasets=DATASETS, top_n=TOP_N_FINE)
    plot_stacked_bars(intent_props_fine,
                      title=f"Intent (Fine, Top {TOP_N_FINE}+Other) Proportions by Dataset",
                      file_basename="intent_fine_stacked")

    # Form: major / fine stacked bars
    form_props_major = build_props_for_form(form_df, level="major", datasets=DATASETS)
    plot_stacked_bars(form_props_major,
                      title="Form (Major) Proportions by Dataset",
                      file_basename="form_major_stacked")

    form_props_fine = build_props_for_form(form_df, level="fine", datasets=DATASETS, top_n=TOP_N_FINE)
    plot_stacked_bars(form_props_fine,
                      title=f"Form (Fine, Top {TOP_N_FINE}+Other) Proportions by Dataset",
                      file_basename="form_fine_stacked")

    # Cosine similarity matrices (major & fine)
    intent_sim_major = cosine_similarity_matrix(intent_props_major)
    form_sim_major   = cosine_similarity_matrix(form_props_major)
    intent_sim_fine  = cosine_similarity_matrix(intent_props_fine)
    form_sim_fine    = cosine_similarity_matrix(form_props_fine)

    # Save similarity CSVs under QuestionType_Analysis/
    # if not intent_sim_major.empty: intent_sim_major.to_csv(OUTPUT_DIR / "intent_similarity_major.csv", encoding="utf-8-sig")
    # if not form_sim_major.empty:   form_sim_major.to_csv(OUTPUT_DIR / "form_similarity_major.csv",   encoding="utf-8-sig")
    # if not intent_sim_fine.empty:  intent_sim_fine.to_csv(OUTPUT_DIR / "intent_similarity_fine.csv", encoding="utf-8-sig")
    # if not form_sim_fine.empty:    form_sim_fine.to_csv(OUTPUT_DIR / "form_similarity_fine.csv",     encoding="utf-8-sig")

    # Heatmaps with a nicer colormap (default 'viridis', configurable via CFG['similarity_cmap'])
    plot_similarity_heatmap(intent_sim_major, "Dataset Similarity (Intent Major, Cosine)", "intent_sim_major")
    plot_similarity_heatmap(form_sim_major,   "Dataset Similarity (Form Major, Cosine)",   "form_sim_major")
    plot_similarity_heatmap(intent_sim_fine,  "Dataset Similarity (Intent Fine, Cosine)",  "intent_sim_fine")
    plot_similarity_heatmap(form_sim_fine,    "Dataset Similarity (Form Fine, Cosine)",    "form_sim_fine")

if __name__ == "__main__":
    main()
