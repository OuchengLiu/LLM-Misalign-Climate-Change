from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import ast
import unicodedata as _ud
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Configuration (loaded once, all UPPER_CASE)                                 #
# --------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parent.parent / "Configs.yaml"
SECTION = "1_RealWorld_Conversations_Extraction"

if not CFG_PATH.is_file():
    sys.exit(f"Config YAML not found: {CFG_PATH}")

with CFG_PATH.open("r", encoding="utf-8") as f:
    ROOT_CFG = yaml.safe_load(f) or {}

CFG = ROOT_CFG.get(SECTION) or {}
CFG = CFG.get("Extraction_by_Keywords", {})
# Required keys with sane defaults / validation
SUBJECT: str = CFG.get("Subject", "Climate change")
NUM_KEYWORDS: int = int(CFG.get("Number_of_Keywords", 20))
LANGUAGES: List[str] = CFG.get(
    "Languages",
    [
        "English",
        "Chinese",
        "Russian",
        "French",
        "Spanish",
        "Portuguese",
        "German",
        "Arabic",
        "Italian",
        "Turkish",
        "Japanese",
        "Korean",
        "Polish",
        "Vietnamese",
    ],
)
if "English" not in LANGUAGES:
    LANGUAGES.insert(0, "English")


GET_KEYWORDS_PROMPT_FILE = Path(CFG.get("Get_Keywords_Prompt", "../Prompts/Get_Keywords.txt"))
TRANSLATE_PROMPT_FILE = Path(CFG.get("Translate_Prompt", "../Prompts/Translate_Keywords.txt"))

DATA_DIR = Path(CFG.get("Data_Dir", "../Data"))
INPUT_FILE = DATA_DIR / CFG.get("Input_File", "WildChat_Firstturn_Dedup.jsonl")
SPLIT_TITLE = CFG.get("Split_Title", "WildChat")
LOGS_DIR = Path(CFG.get("Logs_Dir", "../Logs"))

API_KEY = CFG.get("OpenAI_API_Key", os.getenv("OPENAI_API_KEY", ""))
MODEL: str = CFG.get("Model", "gpt-4o")
TEMPERATURE  = CFG.get("Temperature", 0.0)
CLIENT = OpenAI(api_key=API_KEY)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s.strip())[:40]


def _norm(s: str) -> str:
    """Unicode‑aware, case‑folded normalisation used for dict keys."""
    return _ud.normalize("NFKC", s).casefold()


def make_log_dir(subject: str) -> Tuple[Path, str]:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = _safe_name(subject)
    path = LOGS_DIR / f"{safe_subject}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path, safe_subject


def _load_prompt(path: Path, **kwargs) -> str:
    text = path.read_text(encoding="utf-8")
    return text.format(**kwargs)


def chat_completion(prompt: str) -> str:
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content  # type: ignore

# --------------------------------------------------------------------------- #
# Keyword generation & translation                                            #
# --------------------------------------------------------------------------- #

def strip_code_fences(s: str) -> str:
    lines = s.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def fetch_keywords(subject: str, n: int) -> List[str]:
    prompt = _load_prompt(GET_KEYWORDS_PROMPT_FILE, SUBJECT=subject, N=n)
    raw_reply = chat_completion(prompt)
    cleaned_reply = strip_code_fences(raw_reply)
    try:
        kws = json.loads(cleaned_reply)
        if isinstance(kws, list):
            return [str(k).strip() for k in kws if str(k).strip()]
    except json.JSONDecodeError:
        pass
    kws = re.split(r"[\n,]+", cleaned_reply)
    return [k.strip() for k in kws if k.strip()][:n]


def translate_keywords(keywords: List[str], langs: List[str]) -> Dict[str, List[str]]:
    prompt = _load_prompt(
        TRANSLATE_PROMPT_FILE,
        KEYWORDS=json.dumps(keywords, ensure_ascii=False),
        LANGUAGES=", ".join(langs),
    )
    reply = chat_completion(prompt)
    cleaned = strip_code_fences(reply)

    # Try to parse as JSON first.
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {}  # Fallback if GPT did not output valid JSON

    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for raw_lang, raw_val in data.items():
            # Remove extra quotes around language names.
            lang = raw_lang.strip().strip('"').strip("'")

            # Ignore explanatory keys not present in *langs*.
            if lang not in langs:
                continue

            # raw_val can be a list *or* a string that looks like a list.
            if isinstance(raw_val, str):
                try:
                    raw_val = ast.literal_eval(raw_val)  # Safely convert string → list
                except Exception:
                    raw_val = re.split(r",|\n|;", raw_val)

            # Clean individual keywords.
            keywords_clean = [
                str(x).strip().strip('"').strip("'") for x in raw_val if str(x).strip()
            ]
            out[lang] = keywords_clean

    # Ensure every requested language has a key in the result dict.
    for lang in langs:
        out.setdefault(lang, [])

    return out


# --------------------------------------------------------------------------- #
# Conversation filtering & statistics                                         #
# --------------------------------------------------------------------------- #

def filter_conversations(
    input_file: Path,
    keyword_map: Dict[str, List[str]],
    output_file: Path,
) -> Tuple[int, Dict[str, Dict[str, int]]]:
    counts: Dict[str, Dict[str, int]] = {
        lang: {kw: 0 for kw in kws} for lang, kws in keyword_map.items()
    }

    flat_map: Dict[str, Tuple[str, str]] = {}
    for lang, kws in keyword_map.items():
        for kw in kws:
            flat_map[_norm(kw)] = (lang, kw)

    pattern = re.compile("|".join(map(re.escape, flat_map.values() if False else [kw for _, kw in flat_map.values()])), re.IGNORECASE)

    total_matches = 0
    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text: str = obj.get("text", "")
            if not text:
                continue
            matched_kw_norm = {_norm(m.group(0)) for m in pattern.finditer(text)}
            if matched_kw_norm:
                total_matches += 1
                fout.write(line)
                for kw_norm in matched_kw_norm:
                    info = flat_map.get(kw_norm)
                    if info is None:
                        # keyword variant not in map (likely accent or typo) – skip counting but keep line.
                        continue
                    lang, orig_kw = info
                    counts[lang][orig_kw] += 1
    return total_matches, counts

# --------------------------------------------------------------------------- #
# Reporting helper                                                             #
# --------------------------------------------------------------------------- #

def write_data_info(
    file_path: Path,
    counts: Dict[str, Dict[str, int]],
    total: int,
    subject: str,
) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Total Conversations Extracted: {total}\n\n")
        for lang in counts:
            f.write(f"{lang}\n")
            f.write("-" * len(lang) + "\n")
            for kw, c in sorted(counts[lang].items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{kw}: {c}\n")
            f.write("\n")

# --------------------------------------------------------------------------- #
# Pipeline orchestrator                                                       #
# --------------------------------------------------------------------------- #

def prompt_keyword_edit(keywords: List[str]) -> List[str]:
    print("\nGenerated keywords (→ you can edit):")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:>2}. {kw}")
    print(
        "\nLeave blank to accept, or enter a *comma‑separated* replacement list (whitespace will be trimmed):"
    )
    user_in = input("» ").strip()
    if not user_in:
        return keywords
    edited = [k.strip() for k in re.split(r",|\n", user_in) if k.strip()]
    if not edited:
        print("No valid keywords detected – keeping original list.\n")
        return keywords
    print(f"Using your modified list ({len(edited)} keywords).\n")
    return edited


def run_pipeline() -> None:
    log_dir, safe_subject = make_log_dir(SUBJECT)

    keywords = fetch_keywords(SUBJECT, NUM_KEYWORDS)[:NUM_KEYWORDS]
    keywords = prompt_keyword_edit(keywords)

    translations = translate_keywords(keywords, LANGUAGES)
    keyword_map: Dict[str, List[str]] = {"English": keywords}
    keyword_map.update(translations)

    kw_json_path = log_dir / "Keywords.json"
    with kw_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "Subject": SUBJECT,
                "Number_of_Keywords": len(keywords),
                "Keywords": keywords,
                "Keywords_in_Different_Languages": translations,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    output_conv = log_dir / f"{SPLIT_TITLE}_RealWorld_Conversations.jsonl"
    total_matches, counts = filter_conversations(INPUT_FILE, keyword_map, output_conv)

    data_info_file = log_dir / f"{safe_subject}_Data_Info.txt"
    write_data_info(data_info_file, counts, total_matches, SUBJECT)

    print("\nPipeline complete ✅")
    print(f"Keywords used              : {len(keywords)}")
    print(f"Languages translated       : {len(translations)} (include English)")
    print(f"Conversations extracted    : {total_matches}")
    print("Log directory:")
    print(f"  {log_dir}\n")


if __name__ == "__main__":
    run_pipeline()