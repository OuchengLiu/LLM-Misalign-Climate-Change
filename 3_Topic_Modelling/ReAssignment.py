from __future__ import annotations

import json
import os
import re
import sys
import time
import socket
from pathlib import Path
from typing import List, Set

import unicodedata as ud
import yaml
from tqdm import tqdm

try:
    from openai import OpenAI  # v1 client
    try:
        from openai import APIConnectionError, APITimeoutError  # type: ignore
    except Exception:
        APIConnectionError = tuple()
        APITimeoutError = tuple()
except ImportError:
    OpenAI = None
    APIConnectionError = tuple()
    APITimeoutError = tuple()


# ---------------- Config ----------------

def load_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / "Configs.yaml"
    with cfg_file.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    return cfg.get("3_Topic_Modelling", {}).get("ReAssignment", {})


CFG = load_cfg()

DATA_PATH = Path(CFG.get("data_path", "Logs/Final_Climate Change/All_Data_with_Final_Topic.jsonl")).expanduser().resolve()
PROMPT_PATH = Path(CFG.get("prompt_path", "Prompts/3_Reassign_Topics_ClimateChange.txt")).expanduser().resolve()
OUT_PATH = DATA_PATH.parent / "All_Data_with_Reassigned_Topic.jsonl"

OPENAI_MODEL = CFG.get("openai_model", "gpt-5-mini")
OPENAI_KEY = CFG.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
MAX_RETRIES = int(CFG.get("max_retries", 3))
FLUSH_INTERVAL = int(CFG.get("flush_interval", 5))
TEMPERATURE = float(CFG.get("temperature", 0.2))
CHECK_INTERVAL_SECS = float(CFG.get("connectivity_check_interval_secs", 5.0))
BACKOFF_MAX_SECS = float(CFG.get("connectivity_backoff_max_secs", 30.0))


# ---------------- Connectivity helpers ----------------

def _can_connect(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_openai_reachable():
    delay = CHECK_INTERVAL_SECS
    while not _can_connect("api.openai.com", 443):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[NET] {ts} OpenAI unreachable; retrying in {int(delay)}s…", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay = min(delay * 1.5, BACKOFF_MAX_SECS)


def is_network_exception(exc: Exception) -> bool:
    et = type(exc).__name__
    msg = str(exc).lower()
    network_keywords = (
        "connection", "timeout", "timed out", "network", "temporarily unavailable",
        "reset", "broken pipe", "ssl", "handshake", "eof", "name resolution", "dns"
    )
    return (
        isinstance(exc, (OSError, TimeoutError)) or
        et in {"APIConnectionError", "APITimeoutError"} or
        any(k in msg for k in network_keywords)
    )


def clean(text: str) -> str:
    """Normalize whitespace and unicode control characters."""
    text = ud.normalize("NFKC", text)
    text = "".join(ch for ch in text if ud.category(ch) != "Cf")
    text = text.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()


# ---------------- Fixed taxonomy ----------------
FIXED_TAXONOMY: List[str] = [
    # A. Climate Science Foundations & Methods
    "A1. Atmospheric Science & Climate Processes",
    "A2. Greenhouse Gas & Biogeochemical Cycles",
    "A3. Oceans, Cryosphere & Sea-Level Change",
    "A4. Extreme Weather Events",
    "A5. Climate Modeling",
    "A6. Environmental Monitoring",
    # B. Ecological Impacts
    "B1. Biodiversity Loss",
    "B2. Terrestrial & Freshwater Ecosystem Changes",
    "B3. Marine & Coastal Ecosystem Changes",
    # C. Human Systems & Socioeconomic Impacts
    "C1. Agriculture & Food Security",
    "C2. Water Resources & Hydrological Impacts",
    "C3. Human Health & Well-being",
    "C4. Social Equity, Vulnerability & Migration",
    "C5. Urban Systems & Infrastructure Impacts",
    "C6. Service & Industry Sector Impacts",
    # D. Adaptation Strategies
    "D1. Agricultural & Food System Adaptation",
    "D2. Urban Planning, Adaptation & Resilience",
    "D3. Public Health Adaptation",
    "D4. Public Awareness, Communication & Community Engagement",
    "D5. Natural Resource Management & Conservation",
    # E. Mitigation Mechanisms
    "E1. Climate Policy, Governance & Finance Mechanism",
    "E2. Energy Transition",
    "E3. Corporate & Industry Climate Action",
    "E4. Land Use & Ecosystem-based Mitigation",
    "E5. Transport & Building Emissions Reduction",
    # F. Others
    "F1. Others",
]


# ---------------- OpenAI call ----------------

def call_openai(system_instructions: str, payload: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai>=1.0 package is required")
    client = OpenAI(api_key=OPENAI_KEY)
    attempts = 0
    last = ""
    while True:
        ensure_openai_reachable()
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                instructions=system_instructions,
                input=payload,
                reasoning={"effort": "low"},  # minimal, low, medium, high
                # temperature=TEMPERATURE,
            )
            out = clean((resp.output_text or "").strip())
            if out:
                return out
            attempts += 1
            last = out
        except Exception as e:
            if is_network_exception(e):
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[NET] {ts} OpenAI network error: {e}. Waiting for recovery…", file=sys.stderr, flush=True)
                continue
            attempts += 1
            last = clean(str(e))

        if attempts >= MAX_RETRIES:
            print(f"[RETRY] payload for doc: {payload}; last_output: {last}")
            return last
        else:
            print(f"[RETRY] attempt {attempts} for doc: {payload}; last_output: {last}")


# ---------------- Parsing & validation ----------------

def is_irrelevant_topics(final_topics) -> bool:
    """Return True if Final_Topics contains 'Irrelevant Data' anywhere (case-insensitive)."""
    if isinstance(final_topics, list):
        for v in final_topics:
            if isinstance(v, str) and v.strip().lower() == "irrelevant data":
                return True
    return False


def parse_topics(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list) or not (1 <= len(arr) <= 3):
            return []
        topics: List[str] = []
        for obj in arr:
            if not isinstance(obj, dict) or set(obj.keys()) != {"topic"}:
                return []
            t = str(obj["topic"]).strip()
            if not t:
                return []
            topics.append(t)
        return topics
    except Exception:
        return []


def validate_topics(topics: List[str], allowed: Set[str]) -> bool:
    return all(t in allowed for t in topics)


# ---------------- Main ----------------

def process():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    system_instructions = PROMPT_PATH.read_text(encoding="utf-8").strip()
    allowed_list = FIXED_TAXONOMY
    allowed_set = set(allowed_list)

    # Print taxonomy at startup
    print(f"[TAXONOMY] {len(allowed_list)} topics:")
    for t in allowed_list:
        print(f"  - {t}")

    # Count total and relevant records
    total_count = 0
    irrelevant_count = 0
    with DATA_PATH.open("r", encoding="utf-8") as fscan:
        for line in fscan:
            total_count += 1
            try:
                rec = json.loads(line)
                if is_irrelevant_topics(rec.get("Final_Topics")):
                    irrelevant_count += 1
            except Exception:
                pass
    remaining_count = max(total_count - irrelevant_count, 0)
    print(f"[DATA] total={total_count}, deleted_irrelevant={irrelevant_count}, to_process={remaining_count}")

    processed = 0
    with DATA_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        with tqdm(total=remaining_count, desc="Reassigning topics", unit="doc") as pbar:
            for ln in fin:
                obj = json.loads(ln)

                if obj.get("Final_Topics") == ["Irrelevant Data"]:
                    continue

                doc = {
                    "id": obj.get("id"),
                    "text": (obj.get("text") or "").strip(),
                }
                if not doc["text"]:
                    pbar.update(1)
                    continue

                payload = f"INPUT TEXT: [{{{doc['text']}}}]"

                backoff = 2.0
                topics: List[str] = []
                for attempt in range(1, MAX_RETRIES + 1):
                    raw = call_openai(system_instructions, payload)
                    topics = parse_topics(raw)
                    if topics and validate_topics(topics, allowed_set):
                        break
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 15.0)
                else:
                    topics = ["F1. Others"] if "F1. Others" in allowed_set else [allowed_list[-1]]

                out = {
                    "id": doc["id"],
                    "text": doc["text"],
                    "Final_Topics": topics,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1

                if processed % FLUSH_INTERVAL == 0:
                    fout.flush()
                    try:
                        os.fsync(fout.fileno())
                    except Exception:
                        pass

                pbar.update(1)

    try:
        with OUT_PATH.open("a", encoding="utf-8") as fp:
            fp.flush()
            os.fsync(fp.fileno())
    except Exception:
        pass

    print(f"[DONE] Wrote {processed} records → {OUT_PATH}")


if __name__ == "__main__":
    t0 = time.time()
    try:
        process()
    finally:
        print(f"Elapsed {time.time()-t0:.1f}s")
