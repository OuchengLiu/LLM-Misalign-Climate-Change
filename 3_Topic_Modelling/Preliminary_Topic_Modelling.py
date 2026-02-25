from __future__ import annotations

import csv
import json
import os
import sys
import time
import re
import socket
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import unicodedata as ud
from collections import Counter

# YAML config loader
import yaml
try:
    from openai import OpenAI  # v1 client
    try:
        # These exception classes may not exist in older SDKs; guarded import
        from openai import APIConnectionError, APITimeoutError  # type: ignore
    except Exception:
        APIConnectionError = tuple()  # harmless placeholder
        APITimeoutError = tuple()
except ImportError:
    OpenAI = None  # optional
    APIConnectionError = tuple()
    APITimeoutError = tuple()

# Ollama client (optional)
try:
    from ollama import chat
except ImportError:
    chat = None  # optional

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    """Load and return the 3_Topic_Modelling section from Configs.yaml."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / "Configs.yaml"
    with cfg_file.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    return cfg.get("3_Topic_Modelling", {})

CFG = load_cfg()
MCFG = CFG.get("Preliminary_Topic_Modelling", {})

# Core parameters -----------------------------------------------------------
SUBJECT: str = MCFG.get("subject", "Climate Change")
SUBJECT_SLUG = SUBJECT.replace(" ", "_")
# max words for related_domain
SUB_WORDS: int = int(MCFG.get("sub_words", 4))
# max words for explanation
EXPLANATION_WORDS: int = int(MCFG.get("explanation_words", 15))

DATA_PATH = Path(MCFG.get(
    "data_path",
    f"Logs/{SUBJECT_SLUG}_All_Data.jsonl",
)).expanduser().resolve()
PROMPT_PATH = Path(MCFG.get("prompt_path", "../Prompts/3_Generate_Topics.txt")).expanduser().resolve()
LENGTH_OF_TOPIC = SUB_WORDS + len(SUBJECT) + 2  # for ": " separator
MAX_RETRIES: int = int(MCFG.get("max_retries", 3))
LLM_PROVIDER: str = MCFG.get("llm_provider", "openai").lower()
FLUSH_INTERVAL = int(MCFG.get("flush_interval", 50))

# OpenAI params -------------------------------------------------------------
OPENAI_MODEL = MCFG.get("openai_model", "gpt-4o")
OPENAI_KEY = MCFG.get("openai_api_key", "")
TEMPERATURE = float(MCFG.get("temperature", 0.0))

# Ollama params -------------------------------------------------------------
OLLAMA_MODEL = MCFG.get("ollama_model", "qwen3:30b")
OLLAMA_URL = MCFG.get("ollama_base_url", "http://localhost:11434/api/generate")

# Connectivity wait/backoff -------------------------------------------------
# These control how aggressively we poll for connectivity restoration.
CHECK_INTERVAL_SECS = float(MCFG.get("connectivity_check_interval_secs", 5.0))
BACKOFF_MAX_SECS = float(MCFG.get("connectivity_backoff_max_secs", 30.0))

# ---------------------------------------------------------------------------
# I/O setup
# ---------------------------------------------------------------------------

def ensure_io() -> Tuple[Path, Path, Path]:
    """Ensure input path exists and return data, jsonl-output, csv-output paths."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    log_dir = DATA_PATH.parent
    # JSONL output with topic indices
    out_jsonl = log_dir / "All_Data_with_Topic.jsonl"
    # CSV vocab + explanations
    topics_dir = log_dir / "Topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    topics_csv = topics_dir / "Topics_1.csv"
    return DATA_PATH, out_jsonl, topics_csv

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def load_instructions() -> str:
    """Load prompt template and substitute {subject}, {n}, {m} placeholders."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # Avoid .format because the prompt may contain literal braces.
    instructions = text.replace("{subject}", SUBJECT)
    instructions = instructions.replace("{n}", str(SUB_WORDS))
    instructions = instructions.replace("{m}", str(EXPLANATION_WORDS))
    return instructions

INSTRUCTIONS = load_instructions()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Normalize whitespace and unicode control characters."""
    text = ud.normalize("NFKC", text)
    text = "".join(ch for ch in text if ud.category(ch) != "Cf")
    text = text.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate_related(s: str, max_words: int = SUB_WORDS) -> str:
    """Take at most `max_words` tokens from a space-delimited string."""
    words = s.split()
    return " ".join(words[:max_words])


def strip_markdown_fences(raw: str) -> str:
    """
    Remove common Markdown code fences (``` or ```json ... ```), leaving inner JSON intact.
    This is more conservative than deleting everything between the first and last fence.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)  # strip opening fence
        raw = re.sub(r"\s*```$", "", raw)               # strip closing fence
    return raw.strip()


def parse_topics_and_explanations(raw: str) -> List[Dict[str, str]]:
    """
    Parse the model's JSON output into a list of {topic, explanation} dicts.
    Applies word limits to both the topic's related-domain part and the explanation.
    Returns [] on parse failure so callers can decide how to handle it.
    """
    raw = strip_markdown_fences(raw)
    try:
        arr = json.loads(raw)
        out = []
        for obj in arr:
            topic = (obj.get("topic") or "").strip()
            expl  = (obj.get("explanation") or "").strip()

            # Enforce topic word limit. If the topic looks like "SUBJECT: related",
            # only trim the related part; otherwise trim the whole string.
            if ": " in topic:
                subj, rel = topic.split(": ", 1)
                rel = truncate_related(rel, SUB_WORDS)
                if subj.strip().lower() == SUBJECT.lower() and rel.strip().lower() == "irrelevant data":
                    topic = "Irrelevant Data"
                else:
                    topic = f"{subj}: {rel}"
            else:
                topic = truncate_related(topic, LENGTH_OF_TOPIC)

            # Enforce explanation word limit.
            expl = " ".join(expl.split()[:EXPLANATION_WORDS])

            if topic:
                out.append({"topic": topic, "explanation": expl})
        return out
    except json.JSONDecodeError:
        return []

# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

def _can_connect(host: str, port: int, timeout: float = 3.0) -> bool:
    """Basic TCP connect test to (host, port)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_openai_reachable():
    """
    Block until api.openai.com:443 is reachable.
    This is a proxy for general outbound connectivity.
    """
    delay = CHECK_INTERVAL_SECS
    while not _can_connect("api.openai.com", 443):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[NET] {ts} OpenAI unreachable; retrying in {int(delay)}s…", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay = min(delay * 1.5, BACKOFF_MAX_SECS)
    # print("[NET] OpenAI connectivity restored.", file=sys.stderr, flush=True)


def ensure_ollama_reachable():
    """
    Block until the Ollama host:port derived from OLLAMA_URL is reachable.
    """
    p = urlparse(OLLAMA_URL)
    host = p.hostname or "localhost"
    port = p.port or 11434
    delay = CHECK_INTERVAL_SECS
    while not _can_connect(host, port):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[NET] {ts} Ollama({host}:{port}) unreachable; retrying in {int(delay)}s…", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay = min(delay * 1.5, BACKOFF_MAX_SECS)
    # print("[NET] Ollama connectivity restored.", file=sys.stderr, flush=True)


def is_network_exception(exc: Exception) -> bool:
    """
    Best-effort heuristic to classify an exception as a network/connectivity error.
    This includes socket errors/timeouts and common SDK network error types/phrases.
    """
    et = type(exc).__name__
    msg = str(exc).lower()
    network_keywords = (
       "connection", "timeout", "timed out", "network",
       "temporarily unavailable", "reset", "broken pipe",
       "ssl", "handshake", "eof", "name resolution", "dns"
    )
    return (
        isinstance(exc, (OSError, TimeoutError)) or
        et in {"APIConnectionError", "APITimeoutError"} or
        any(k in msg for k in network_keywords)
    )

# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------

def llm_openai(text: str) -> str:
    """
    Call OpenAI Responses API with "instructions" + "input".
    - If connectivity is down, block until it returns (doesn't eat into MAX_RETRIES).
    - Non-network exceptions or parse failures count toward MAX_RETRIES.
    """
    if OpenAI is None:
        raise RuntimeError("openai package >= 1.0 not installed")

    client = OpenAI(api_key=OPENAI_KEY or os.getenv("OPENAI_API_KEY", ""))
    attempts = 0
    last = ""
    while True:
        ensure_openai_reachable()
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                instructions=INSTRUCTIONS,
                input=f"Text Needs to be Identified: {text}",
                temperature=TEMPERATURE,
                # reasoning={"effort": "minimal"}
            )
            out = clean(resp.output_text)

            if parse_topics_and_explanations(out):
                return out
            # JSON parse failed → count as a retriable attempt
            attempts += 1
            last = out
        except Exception as e:
            if is_network_exception(e):
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[NET] {ts} OpenAI network error: {e}. Waiting for recovery…", file=sys.stderr, flush=True)
                continue  # do NOT consume MAX_RETRIES
            # Non-network error → consume retry
            attempts += 1
            last = clean(str(e))

        if attempts >= MAX_RETRIES:
            return last


def llm_ollama(text: str) -> str:
    """
    Call Ollama chat API.
    - If connectivity is down, block until it returns (doesn't eat into MAX_RETRIES).
    - Non-network exceptions or parse failures count toward MAX_RETRIES.
    """
    if chat is None:
        raise RuntimeError("ollama package not installed")
    attempts = 0
    last = ""
    while True:
        ensure_ollama_reachable()
        try:
            resp = chat(
                model=OLLAMA_MODEL,
                messages=[{"role":"user","content":INSTRUCTIONS + "\n\nText: " + text}],
                stream=False,
                options={"temperature": TEMPERATURE}
            )
            last = clean(resp["message"]["content"])
            if parse_topics_and_explanations(last):
                return last
            attempts += 1  # parse failure
        except Exception as e:
            if is_network_exception(e):
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[NET] {ts} Ollama network/connection error: {e}. Waiting for recovery…", file=sys.stderr, flush=True)
                continue  # do NOT consume MAX_RETRIES
            attempts += 1
            last = clean(str(e))

        if attempts >= MAX_RETRIES:
            return last


def call_llm(text: str) -> str:
    """Dispatch to the configured provider."""
    if LLM_PROVIDER == "openai":
        return llm_openai(text)
    if LLM_PROVIDER == "ollama":
        return llm_ollama(text)
    raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")

# ---------------------------------------------------------------------------
# CSV helpers (atomic write + fsync)
# ---------------------------------------------------------------------------

def load_topic_vocab(csv_path: Path) -> Tuple[Dict[str, Tuple[str,int]], List[str]]:
    """
    Load CSV to a vocabulary dict and an order list.

    Returns:
      vocab: {topic: (explanation, count)}
      order: [topic1, topic2, ...]  (stable order for indices)
    """
    vocab, order = {}, []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as fp:
            for row in csv.reader(fp):
                if not row:
                    continue
                topic = row[0]
                expl  = row[1] if len(row) > 1 else ""
                try:
                    count = int(row[2]) if len(row) > 2 else 0
                except Exception:
                    count = 0
                vocab[topic] = (expl, count)
                order.append(topic)
    return vocab, order


def _fsync_file(path: Path):
    """
    Attempt to fsync the file handle to ensure durability on supported platforms.
    If not supported, silently degrade to normal flush semantics.
    """
    try:
        with path.open("a", encoding="utf-8") as fp:
            fp.flush()
            os.fsync(fp.fileno())
    except Exception:
        pass


def flush_vocab(csv_path: Path, vocab: Dict[str, Tuple[str,int]], order: List[str]):
    """
    Atomically write the CSV:
      - Write to a temporary file.
      - Flush + fsync the temp file.
      - os.replace to atomically swap into place.
      - fsync the final path as well.
    """
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        for t in order:
            expl, cnt = vocab[t]
            writer.writerow([t, expl, cnt])
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, csv_path)
    _fsync_file(csv_path)

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process():
    DATA_PATH, out_jsonl, topics_csv = ensure_io()
    vocab, order = load_topic_vocab(topics_csv)
    processed = 0
    existing = set()

    # Avoid duplicating docs already present in out_jsonl
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as fp:
            for ln in fp:
                try:
                    existing.add(json.loads(ln).get("id"))
                except Exception:
                    pass

    total = sum(1 for _ in DATA_PATH.open("r", encoding="utf-8"))
    with DATA_PATH.open("r", encoding="utf-8") as fin, \
         out_jsonl.open("a", encoding="utf-8") as fout:

        for ln in tqdm(fin, total=total, desc="Processing texts"):
            doc = json.loads(ln)
            if doc.get("id") in existing:
                continue
            text = (doc.get("text", "") or "").strip()
            if not text:
                continue

            try:
                completion = call_llm(text)
                entries = parse_topics_and_explanations(completion)
                if not entries:
                    raise ValueError("Empty parse result")
            except Exception as exc:
                # Fallback topic if parsing or upstream call fails
                print(f"[WARN] id={doc.get('id')} failed to parse topics: {exc}. Response: {completion}", file=sys.stderr)
                entries = [{"topic": "Irrelevant Data", "explanation": ""}]

            # Assign indices; update vocab counts (keep earliest explanation by default)
            indices = []
            for entry in entries:
                topic = entry.get("topic") or "Irrelevant Data"
                expl = entry.get("explanation") or ""
                if topic not in vocab:
                    vocab[topic] = (expl, 0)
                    order.append(topic)
                prev_expl, cnt = vocab[topic]
                # Keep the earliest non-empty explanation; change to (expl or prev_expl) to prefer newest
                vocab[topic] = (prev_expl or expl, cnt + 1)
                indices.append(order.index(topic) + 1)

            doc["Initial_Topic"] = indices
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed += 1

            # Durable, externally visible flush every FLUSH_INTERVAL
            if processed % FLUSH_INTERVAL == 0:
                # 1) CSV: atomic write + fsync
                flush_vocab(topics_csv, vocab, order)
                # 2) JSONL: flush + fsync
                fout.flush()
                try:
                    os.fsync(fout.fileno())
                except Exception:
                    pass
                # 3) (Optional) fsync the directory containing the CSV for extra durability
                try:
                    os.fsync(os.open(str(topics_csv.parent), os.O_RDONLY))
                except Exception:
                    pass

    # Final flush on exit
    flush_vocab(topics_csv, vocab, order)
    try:
        with out_jsonl.open("a", encoding="utf-8") as fp:
            fp.flush()
            os.fsync(fp.fileno())
    except Exception:
        pass

    print(f"[DONE] Processed {processed} new texts → {topics_csv}")

if __name__ == "__main__":
    start = time.time()
    try:
        process()
    finally:
        print(f"Elapsed {time.time()-start:.1f}s")
