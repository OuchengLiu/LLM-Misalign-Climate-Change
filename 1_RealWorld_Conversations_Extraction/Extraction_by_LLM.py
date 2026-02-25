from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata as ud
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import requests
import yaml
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse

# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #
def load_cfg(section: str = "1_RealWorld_Conversations_Extraction") -> dict:
    """Return the requested section from Configs.yaml (repo root assumed)."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / "Configs.yaml"
    with cfg_file.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp) or {}
    return raw_cfg.get(section, {})


CFG = load_cfg()
MCFG = CFG.get("Extraction_by_LLM", {})

# Core parameters ------------------------------------------------------------
RAW_SUBJECTS = MCFG.get("Subject", "Climate Change")
if isinstance(RAW_SUBJECTS, str):
    SUBJECT_LIST = [s.strip() for s in RAW_SUBJECTS.split(",") if s.strip()]
else:  # list in YAML
    SUBJECT_LIST = [s.strip() for s in RAW_SUBJECTS if s.strip()]

DATA_DIR = Path(MCFG.get("Data_Dir", "../Data")).expanduser().resolve()
INPUT_FILE = DATA_DIR / MCFG.get("Input_File", "WildChat_Firstturn_Dedup.jsonl")
OUTPUT_FILE = MCFG.get("Output_File", "WildChat_Firstturn_Dedup_with_Subject.jsonl")
SPLIT_TITLE = MCFG.get("Split_Title", "WildChat")
LOGS_DIR = Path(MCFG.get("Logs_Dir", "../Logs")).expanduser().resolve()
PROMPT_PATH = Path(MCFG.get("Prompt", "../Prompts/1_Subject_Match.txt")).expanduser()

# Ollama parameters ----------------------------------------------------------
OLLAMA_MODEL = MCFG.get("Ollama_Model", "llama3.1:8b")
OLLAMA_URL = MCFG.get("Ollama_Base_Url", "http://localhost:11434/api/generate")
TEMPERATURE = float(MCFG.get("Temperature", 0.0))

# Runtime tuning -------------------------------------------------------------
FLUSH_INTERVAL = int(MCFG.get("Flush_Interval", 100))

# --------------------------------------------------------------------------- #
# Text cleaning                                                               #
# --------------------------------------------------------------------------- #
def clean(text: str) -> str:
    """Normalize & strip hidden Unicode so plain search always works."""
    text = ud.normalize("NFKC", text)
    text = "".join(ch for ch in text if ud.category(ch) != "Cf")
    text = text.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
def sanitize(name: str) -> str:
    """File-system-safe subject slug."""
    return re.sub(r"\s+", "_", re.sub(r"[^\w\s\-]", "", name.strip()))


def read_prompt(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def classify_subject(message: str, subjects: list[str], prompt_tpl: str) -> str | None:
    prompt = (
        prompt_tpl
        .replace("{SUBJECTS}", ", ".join(subjects))
        .replace("{TEXT}", message)
        .strip()
    )
    try:
        resp = chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": TEMPERATURE, "enable_thinking": False, },
        )
        answer_raw = resp["message"]["content"]
        # print(answer_raw)
    except Exception as exc:
        print(f"[WARN] Ollama chat failed: {exc}", file=sys.stderr)
        return None

    answer = clean(answer_raw.splitlines()[-1].rstrip("."))

    for subj in subjects:
        if answer.lower() == subj.lower():
            return subj
    return None if answer.lower() == "none" else None


# --------------------------------------------------------------------------- #
# Main routine                                                                #
# --------------------------------------------------------------------------- #
def run(extra_subjects: Iterable[str], flush_every: int) -> None:
    # Build subject list (YAML + CLI)
    subjects = list(dict.fromkeys(SUBJECT_LIST + list(extra_subjects)))
    # subjects = list(dict.fromkeys([s.strip() for s in subjects if s.strip()]))
    if not subjects:
        print("ERROR: no subjects supplied.", file=sys.stderr)
        sys.exit(1)

    prompt_tpl = read_prompt(PROMPT_PATH)

    # Log folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_log_dir = LOGS_DIR / timestamp
    daily_log_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Path] = {}

    def handle_for(subj: str):
        if subj not in handles:
            folder = daily_log_dir / sanitize(subj)
            folder.mkdir(parents=True, exist_ok=True)
            fp = folder / f"{SPLIT_TITLE}_RealWorld_Conversations.jsonl"
            handles[subj] = fp.open("a", encoding="utf-8")
        return handles[subj]

    output_file = INPUT_FILE.with_name(OUTPUT_FILE)
    out_main = output_file.open("a", encoding="utf-8")

    # Step 1 – count lines
    with INPUT_FILE.open("r", encoding="utf-8") as fin:
        total = sum(1 for _ in tqdm(fin, desc="Counting lines"))

    # Step 2 – classify
    processed = 0
    with INPUT_FILE.open("r", encoding="utf-8") as fin, tqdm(
        total=total, desc="Classifying", unit="line"
    ) as bar:
        for raw in fin:
            bar.update(1)
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                print("[WARN] bad JSON skipped.", file=sys.stderr)
                continue

            text = clean(obj.get("text", ""))
            obj["text"] = text
            subj = classify_subject(text, subjects, prompt_tpl)
            obj["Subject"] = subj

            line_out = json.dumps(obj, ensure_ascii=False)
            out_main.write(line_out + "\n")
            if subj:
                handle_for(subj).write(line_out + "\n")

            processed += 1
            if processed % flush_every == 0:
                out_main.flush()
                for h in handles.values():
                    h.flush()

    # Step 3 – close files
    with tqdm(total=len(handles), desc="Closing files") as bar_c:
        out_main.flush()
        out_main.close()
        for h in handles.values():
            h.flush()
            h.close()
            bar_c.update(1)

    print(f"Done! {processed} lines processed.")
    print(f"Main output  : {output_file}")
    print(f"Subject logs : {LOGS_DIR}/<Subject>_{timestamp}/")


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify conversations via local Ollama.")
    parser.add_argument(
        "-s", "--subjects", default="", help="Extra subjects (comma-separated)"
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=FLUSH_INTERVAL,
        help=f"Flush files every N lines (default from YAML: {FLUSH_INTERVAL})",
    )
    args = parser.parse_args()
    extras = [s.strip() for s in args.subjects.split(",") if s.strip()]
    run(extra_subjects=extras, flush_every=args.flush_every)
