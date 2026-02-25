from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List
import tiktoken
import yaml

###############################################################################
# 0. Configuration (loaded once)
###############################################################################
MAX_TEXT_LENGTH = 50000

CFG_PATH: Path = Path(__file__).resolve().parents[1] / "Configs.yaml"
SECTION = "2_Data_Formats_Unification"

if not CFG_PATH.is_file():
    sys.exit(f"Config YAML not found: {CFG_PATH}")

with CFG_PATH.open("r", encoding="utf-8") as fh:
    ROOT_CFG = yaml.safe_load(fh) or {}

CFG: Dict = ROOT_CFG.get(SECTION, {})

# ---------- basic settings -----------
SUBJECT: str = CFG.get("Subject", "Climate Change")
SUBJECT_SAFE: str = SUBJECT.replace(" ", "_")

# ---------- benchmark -----------
BENCH_ROOT_RAW: str = CFG.get("benchmark_root_path", "./Data/{subject}")
BENCHMARK_ROOT_PATH: Path = Path(
    BENCH_ROOT_RAW.format(subject=SUBJECT_SAFE)
).expanduser().resolve()
BENCHMARK_DATA_LIST: List[str] = CFG.get("benchmark_data_list", [])

# ---------- real world -----------
RW_PATH_RAW: str | None = CFG.get("real_world_data_path")
REAL_WORLD_DATA_PATH: Path | None = (
    Path(RW_PATH_RAW.format(subject=SUBJECT_SAFE)).expanduser().resolve()
    if RW_PATH_RAW else None
)

# ---------- output -----------
SAVE_PATH_RAW: str = CFG.get("save_path", "./Logs/{subject}/All_Data.jsonl")
OUTPUT_FILE: Path = Path(
    SAVE_PATH_RAW.format(subject=SUBJECT_SAFE)
).expanduser().resolve()

# ---------- API keys (optional) -----------
API_KEYS: Dict[str, str] = {k: str(v) for k, v in (CFG.get("api_keys") or {}).items()}

###############################################################################
# 1. Helpers
###############################################################################

def save_jsonl(records: List[dict], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fp:
        for rec in records:
            json.dump(rec, fp, ensure_ascii=False)
            fp.write("\n")


def _(msg: str) -> str:
    return msg

###############################################################################
# 2. Dataset processors
###############################################################################

def process_climaqa_gold(path: Path) -> List[dict]:
    mapping = {
        "cloze_benchmark.csv": "cloze",
        "ffq_benchmark.csv": "ffq",
        "mcq_benchmark.csv": "mcq",
    }
    return _read_climaqa_variant(path, "ClimaQA_Gold", mapping)


def process_climaqa_silver(path: Path) -> List[dict]:
    mapping = {
        "cloze_benchmark_silver.csv": "cloze",
        "ffq_benchmark_silver.csv": "ffq",
        "mcq_benchmark_silver.csv": "mcq",
    }
    return _read_climaqa_variant(path, "ClimaQA_Silver", mapping)


def _read_climaqa_variant(path: Path, prefix: str, mapping: Dict[str, str]) -> List[dict]:
    rows: List[dict] = []
    for csv_name, tag in mapping.items():
        fp = path / csv_name
        if not fp.exists():
            print(_(f"[WARN] Missing file {fp}"))
            continue
        with fp.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) < 2:
                    continue
                rid, q = r[0].strip(), r[1].strip()
                if len(q.split()) > MAX_TEXT_LENGTH:
                    continue
                rows.append({"id": f"{prefix}_{tag}_{rid}", "text": q})
    print(_(f"[INFO] {prefix}: {len(rows)} rows extracted."))
    return rows


def process_climate_fever(path: Path) -> List[dict]:
    fp = (path / "climate-fever-dataset-r1.jsonl") if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] Climate_FEVER file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            claim_id = str(obj.get("claim_id", ln)).strip()
            claim_txt = str(obj.get("claim", "")).strip()
            if not claim_txt or len(claim_txt.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"Climate_FEVER_{claim_id}", "text": claim_txt})
    print(_(f"[INFO] Climate_FEVER: {len(recs)} rows extracted."))
    return recs


def process_environmental_claims(path: Path) -> List[dict]:
    splits = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    recs: List[dict] = []
    row_no = 1
    for sp in splits:
        fp = path / sp if path.is_dir() else path
        if not fp.exists():
            print(_(f"[WARN] Missing Environmental_Claims split: {fp}"))
            continue
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = str(obj.get("text", "")).strip()
                if not txt or len(txt.split()) > MAX_TEXT_LENGTH:
                    continue
                recs.append({"id": f"Environmental_Claims_{row_no}", "text": txt})
                row_no += 1
    print(_(f"[INFO] Environmental_Claims: {len(recs)} rows extracted."))
    return recs


def process_scidcc(path: Path) -> List[dict]:
    fp = (path / "SciDCC.csv") if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] SciDCC file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, 1):
            parts = [
                str(row.get("Category", "")).strip(),
                str(row.get("Title", "")).strip(),
                str(row.get("Summary", "")).strip(),
                str(row.get("Body", "")).strip(),
            ]
            combined = ". ".join(p for p in parts if p)
            if not combined or len(combined.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"SciDCC_{idx}", "text": combined})
    print(_(f"[INFO] SciDCC: {len(recs)} rows extracted."))
    return recs


def process_climateqa(path: Path) -> List[dict]:
    fp = path / "ClimateQ&A.jsonl" if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] ClimateQ&A file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(obj.get("question", "")).strip()
            if not q or len(q.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"ClimateQ&A_{idx}", "text": q})
    print(_(f"[INFO] ClimateQ&A: {len(recs)} rows extracted."))
    return recs


def process_climsight(path: Path) -> List[dict]:
    fp = path / "ClimSight_QA.jsonl" if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] ClimSight file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            theme = str(obj.get("theme", "")).strip()
            question = str(obj.get("question", "")).strip()
            text = f"{theme} {question}".strip()
            if not text or len(text.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"ClimSight_{idx}", "text": text})
    print(_(f"[INFO] ClimSight: {len(recs)} rows extracted."))
    return recs


def process_ipcc_ar6(path: Path) -> List[dict]:
    fp = path / "All_IPCC_AR6_Paragraphs.json" if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] IPCC_AR6 file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            print(_("[WARN] Invalid JSON in IPCC_AR6 file."))
            return recs
        for obj in data:
            wg = str(obj.get("wg", "")).strip()
            chap = str(obj.get("chapter", "")).strip()
            pid = obj.get("para_id")
            text = str(obj.get("text", "")).strip()
            if not text or len(text.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"IPCC_AR6_{wg}_{chap}_{pid}", "text": text})
    print(_(f"[INFO] IPCC_AR6: {len(recs)} rows extracted."))
    return recs


def process_reddit(path: Path) -> List[dict]:
    fp = path / "Reddit_ClimateChange_Questions.jsonl" if path.is_dir() else path
    recs: List[dict] = []
    if not fp.exists():
        print(_(f"[WARN] Reddit file missing: {fp}"))
        return recs
    with fp.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = str(obj.get("title", "")).strip()
            desc = str(obj.get("description", "")).strip()
            text = f"{title} {desc}".strip()
            if not text or len(text.split()) > MAX_TEXT_LENGTH:
                continue
            recs.append({"id": f"Reddit_{idx}", "text": text})
    print(_(f"[INFO] Reddit: {len(recs)} rows extracted."))
    return recs


def process_realworld_conversations(file_path: Path, prefix: str) -> List[dict]:
    """
    for realworld conversations：'WildChat' or 'LMSYSChat'
    """
    recs: List[dict] = []
    if not file_path.exists():
        print(_(f"[WARN] Real-world log not found: {file_path}"))
        return recs
    with file_path.open("r", encoding="utf-8") as fp:
        for ln, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                print(_(f"[WARN] Line {ln} invalid JSON ({err}); skipped."))
                continue
            h = str(obj.get("conversation_hash", "")).strip()
            txt = str(obj.get("text", "")).strip()
            if h and txt and len(txt.split()) <= MAX_TEXT_LENGTH:
                # for LMSYSChat，if hash empty, using line number ln
                if prefix == "LMSYSChat":
                    id_str = f"{prefix}_{ln}"
                else:
                    id_str = f"{prefix}_{h}"
                recs.append({"id": id_str, "text": txt})
    print(_(f"[INFO] {prefix}: {len(recs)} rows extracted."))
    return recs

###############################################################################
# 3. Main aggregation
###############################################################################
PROCESSOR_REGISTRY: Dict[str, Callable[[Path], List[dict]]] = {
    "ClimaQA_Gold":         process_climaqa_gold,
    "ClimaQA_Silver":       process_climaqa_silver,
    "Climate_FEVER":        process_climate_fever,
    "Environmental_Claims": process_environmental_claims,
    "SciDCC":               process_scidcc,
    "ClimateQ&A":           process_climateqa,
    "ClimSight":            process_climsight,
    "IPCC_AR6":             process_ipcc_ar6,
    "Reddit":               process_reddit,
}


def unify_all(dry_run: bool = False) -> None:
    records: List[dict] = []
    counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}

    # 3-1 ▸ Benchmark datasets
    for ds_name in BENCHMARK_DATA_LIST:
        ds_path = BENCHMARK_ROOT_PATH / ds_name
        proc = PROCESSOR_REGISTRY.get(ds_name)
        if proc is None:
            print(_(f"[WARN] No processor for {ds_name}; skipped."))
            continue
        recs = proc(ds_path)
        counts[ds_name] = len(recs)
        token_counts[ds_name] = sum(count_tokens(r["text"]) for r in recs)
        records.extend(recs)
        print(_(f"[INFO] ➕ Added {len(recs)} rows from {ds_name} → All_Data."))

    # 3-2 ▸ Real-world WildChat & LMSYSChat logs
    if REAL_WORLD_DATA_PATH is not None:
        rw_records: List[dict] = []

        if REAL_WORLD_DATA_PATH.is_dir():
            filenames = [
                "WildChat_RealWorld_Conversations.jsonl",
                "LMSYSChat_RealWorld_Conversations.jsonl",
            ]
            for fname in filenames:
                path = REAL_WORLD_DATA_PATH / fname
                prefix = "WildChat" if fname.startswith("WildChat") else "LMSYSChat"
                recs = process_realworld_conversations(path, prefix)
                counts[prefix] = len(recs)
                token_counts[prefix] = sum(len(r["text"].split()) for r in recs)
                rw_records.extend(recs)
                print(_(f"[INFO] ➕ Added {len(recs)} rows from {prefix} → All_Data."))
        else:
            for prefix in ["WildChat", "LMSYSChat"]:
                recs = process_realworld_conversations(REAL_WORLD_DATA_PATH, prefix)
                counts[prefix] = len(recs)
                token_counts[prefix] = sum(len(r["text"].split()) for r in recs)
                rw_records.extend(recs)
                print(_(f"[INFO] ➕ Added {len(recs)} rows from {prefix} → All_Data."))

        records = rw_records + records

    # 3-3 ▸ Persist unified file
    total = len(records)
    if not total:
        sys.exit("[ERR] No data collected – aborting.")

    if dry_run:
        print(_(f"[DRY-RUN] Would write {total} rows to {OUTPUT_FILE}"))
        return

    save_jsonl(records, OUTPUT_FILE)
    print(_(f"[INFO] ✔ All_Data.jsonl saved to: {OUTPUT_FILE}  (rows: {total})"))

    # 3-4 ▸ Generate data conclusion
    summary_fp = OUTPUT_FILE.parent / "Data_Conclusion.txt"
    with summary_fp.open("w", encoding="utf-8") as sf:
        for ds, cnt in counts.items():
            sf.write(f"{ds}: {cnt} records, {token_counts.get(ds, 0)} tokens\n")
        sf.write(f"Total records: {total}\n")
        sf.write(f"Total tokens: {sum(token_counts.values())}\n")
    print(_(f"[INFO] Data_Conclusion.txt saved to: {summary_fp}"))


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


###############################################################################
# 4. CLI
###############################################################################

def main() -> None:
    p = argparse.ArgumentParser(description="Unify datasets → All\Data.jsonl")
    p.add_argument("--dry-run", action="store_true", help="Run without writing output file")
    args = p.parse_args()
    unify_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
