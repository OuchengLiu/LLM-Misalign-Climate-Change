import os
import re
import json
import time
import yaml
import typing as T
import requests
from pathlib import Path
from tqdm import tqdm
import unicodedata as ud
import socket
import sys

# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #
def load_cfg(section: str = "4_Type_Modelling") -> dict:
    """Load Configs.yaml and return the nested section for Type Classification."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / "Configs.yaml"
    with cfg_file.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp) or {}
    return raw_cfg.get(section, {}).get("Type_Classification_ClimateChange", {})

CFG = load_cfg()

# --- Parameters from YAML ---
MAX_RETRIES = int(CFG.get("max_retries", 3))
LLM_PROVIDER = CFG.get("llm_provider", "openai")
FLUSH_INTERVAL = int(CFG.get("flush_interval", 20))
INPUT_FILE = Path(CFG.get("data_path", "")).expanduser().resolve()
PROMPT_PATH = Path(CFG.get("prompt_path", "")).expanduser().resolve()

# OpenAI: model & temperature from YAML; API key from environment
OPENAI_KEY = CFG.get("openai_api_key", "")
OPENAI_MODEL = CFG.get("openai_model", "gpt-4.1-mini")
TEMPERATURE = float(CFG.get("temperature", 0.2))

# Ollama settings from YAML
OLLAMA_MODEL = CFG.get("ollama_model", "qwen3:30b")
OLLAMA_BASE_URL = CFG.get("ollama_base_url", "http://localhost:11434/api/generate")

# Optional: keep_alive for Ollama (can be made configurable if needed)
OLLAMA_KEEP_ALIVE = "5m"


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> T.Iterator[dict]:
    """Stream JSONL records with a lenient fallback for trailing commas."""
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield json.loads(line.rstrip(","))
                except Exception:
                    raise


def write_jsonl(path: Path, records: T.List[dict], mode: str = "a") -> None:
    """Append a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_parse_json(text: str) -> dict:
    """
    Parse model output that MUST be JSON; if extra tokens exist,
    try to extract the outermost {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", text.strip())
        if m:
            return json.loads(m.group(0))
        raise


# --------------------------------------------------------------------------- #
# Network helpers                                                             #
# --------------------------------------------------------------------------- #
def is_network_exception(exc: Exception) -> bool:
    """
    Best-effort heuristic to classify an exception as a network/connectivity error.
    Covers socket errors/timeouts and common SDK network error types/phrases.
    """
    et = type(exc).__name__
    msg = str(exc).lower()
    network_keywords = (
        "connection", "timeout", "timed out", "network",
        "temporarily unavailable", "reset", "broken pipe",
        "ssl", "handshake", "eof", "name resolution", "dns",
        "proxy", "refused", "unreachable"
    )
    return (
        isinstance(exc, (OSError, TimeoutError, socket.error)) or
        et in {"APIConnectionError", "APITimeoutError"} or
        any(k in msg for k in network_keywords)
    )

def ensure_openai_reachable(host: str = "api.openai.com", port: int = 443,
                            delay: float = 2.0, backoff: float = 1.5, max_delay: float = 60.0) -> None:
    """
    Block until the OpenAI endpoint is reachable.
    Uses exponential backoff between attempts, capped by max_delay.
    """
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[NET] {ts} OpenAI unreachable; retrying in {int(delay)}s…",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * backoff, max_delay)


# --------------------------------------------------------------------------- #
# Dataset detection (for progress display only)                               #
# --------------------------------------------------------------------------- #
KNOWN_PREFIXES_FOR_DISPLAY = (
    "ClimaQA_Gold",
    "ClimaQA_Silver",
    "Climate_FEVER",
    "ClimateQ&A",
    "ClimateQ\\&A",
    "ClimSight",
    "Environmental_Claims",
    "IPCC_AR6",
    "Reddit",
    "SciDCC",
    "WildChat",
    "LMSYSChat",
)

def detect_dataset(record_id: str) -> str:
    """Return dataset name derived from the id prefix for progress visibility."""
    if not record_id:
        return "Unknown"
    for prefix in KNOWN_PREFIXES_FOR_DISPLAY:
        if record_id.startswith(prefix):
            return "ClimateQ&A" if prefix == "ClimateQ\\&A" else prefix
    return "Other"


# --------------------------------------------------------------------------- #
# Prefix-based deterministic mapping (only for agreed prefixes)               #
# --------------------------------------------------------------------------- #
def prefix_mapping(record_id: str) -> T.Optional[dict]:
    """
    Return a fixed {"intent": [...], "form": [...]} for ONLY the agreed prefixes.
    NOTE: ClimaQA_Gold/Silver is handled specially in main(): Intent by LLM, Form forced.
    """
    rid = record_id or ""
    rid_low = rid.lower()

    # Climate_FEVER — fact-checking claims
    if rid.startswith("Climate_FEVER"):
        return {
            "intent": ["INTENT_1c. Clarification / Verification"],
            "form": ["FORM_7b. Yes/No / True/False"],
        }

    # IPCC_AR6 — Others
    if rid.startswith("IPCC_AR6"):
        return {
            "intent": ["INTENT_9z. Others"],
            "form": ["FORM_9z. Others"],
        }

    # SciDCC — information extraction / labeling
    if rid.startswith("SciDCC"):
        return {
            "intent": ["INTENT_4e. Information Extraction"],
            "form": ["FORM_4b. JSON"],
        }

    # Environmental_Claims — binary sentence classification
    if rid.startswith("Environmental_Claims"):
        return {
            "intent": ["INTENT_4e. Information Extraction"],
            "form": ["FORM_7b. Yes/No / True/False"],
        }

    # Any other dataset: no deterministic mapping → use LLM
    return None


def clean_text(text: str) -> str:
    """Normalize whitespace and unicode."""
    text = ud.normalize("NFKC", text)
    text = "".join(ch for ch in text if ud.category(ch) != "Cf")
    text = text.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()


# -------------------------- ClimaQA helpers -------------------------------- #
def _climaqa_subtype_and_form(record_id: str) -> T.Tuple[str, str]:
    """
    For ClimaQA_Gold/Silver, detect subtype and return:
      (english_question_type_label, forced_form_fullname)
    english_question_type_label is used ONLY to prefix the text sent to LLM
    (e.g., "Free Form Question: {text}"). The forced_form_fullname is used to
    overwrite the LLM's form in final output (taxonomy-complete).
    """
    rid_low = (record_id or "").lower()
    if "_cloze" in rid_low:
        return "Cloze Question", "FORM_1a. Concise Value(s) / Entity(ies)"
    if "_ffq" in rid_low:
        return "Free Form Question", "FORM_1b. Brief Statement"
    if "_mcq" in rid_low:
        return "Multiple Choice Question", "FORM_7a. Multiple Choice"
    # default fallback
    return "Free Form Question", "FORM_1b. Brief Statement"


# --------------------------------------------------------------------------- #
# LLM clients                                                                 #
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """OpenAI client using the Responses API."""

    def __init__(self, model: str, api_key: str, temperature: float, system_prompt: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

    def classify(self, query_text: str, max_retries: int = 3) -> dict:
        """
        Return {"intent": [...], "form": [...]} with taxonomy-complete labels.
        The prompt should define strict taxonomy (INTENT_xx. Name / FORM_xx. Name).:contentReference[oaicite:1]{index=1}
        """
        instructions = self.system_prompt
        input_payload = f"Q: {query_text}\n\nReturn ONLY JSON with keys \"intent\" and \"form\"."

        attempts = 0
        last_err = None

        while True:
            ensure_openai_reachable()
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=input_payload,
                    reasoning={"effort": "low"}
                )
                raw_text = clean_text(resp.output_text)
                data = safe_parse_json(raw_text)
                intent = data.get("intent", [])
                form = data.get("form", [])
                if not isinstance(intent, list):
                    intent = [intent]
                if not isinstance(form, list):
                    form = [form]
                return {"intent": intent, "form": form}
            except Exception as e:
                if is_network_exception(e):
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[NET] {ts} OpenAI network error: {e}. Waiting for recovery…",
                          file=sys.stderr, flush=True)
                    continue
                last_err = e
                attempts += 1
                time.sleep(min(2 ** attempts, 8))
                if attempts >= max_retries:
                    return {"intent": ["INTENT_9z. Others"], "form": ["FORM_9z. Others"], "_error": str(last_err)}


class OllamaClient:
    """Ollama /api/generate client."""

    def __init__(self, model: str, base_url: str, temperature: float, system_prompt: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.system_prompt = system_prompt

    def _post(self, payload: dict) -> dict:
        r = requests.post(self.base_url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def classify(self, query_text: str, max_retries: int = 3) -> dict:
        user_msg = (
            'Q: ' + query_text +
            '\n\nReturn ONLY JSON with keys "intent" and "form".'
        )
        payload = {
            "model": self.model,
            "prompt": user_msg,
            "system": self.system_prompt,
            "options": {"temperature": self.temperature},
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }

        last_err = None
        for attempt in range(max_retries):
            try:
                data = self._post(payload)
                text = data.get("response", "")
                result = safe_parse_json(text)
                intent = result.get("intent", [])
                form = result.get("form", [])
                if not isinstance(intent, list):
                    intent = [intent]
                if not isinstance(form, list):
                    form = [form]
                return {"intent": intent, "form": form}
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Ollama classify failed after {max_retries} tries: {last_err}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    # Validate files
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    # Prepare output path next to input file; OVERWRITE on every run
    out_path = INPUT_FILE.parent / "All_Data_with_Reassigned_Topic_with_QuestionType.jsonl"
    if out_path.exists():
        print(f"[Info] Overwriting existing output: {out_path}")
        out_path.unlink()

    # Load and pre-filter records
    all_records = list(read_jsonl(INPUT_FILE))
    irrelevant_records = [r for r in all_records if "Irrelevant Data" in (r.get("Final_Topics") or [])]
    records = [r for r in all_records if "Irrelevant Data" not in (r.get("Final_Topics") or [])]
    print(f"[Info] Skipped {len(irrelevant_records)} records with Final_Topics containing 'Irrelevant Data'")

    # Init LLM client (created up-front)
    provider = (LLM_PROVIDER or "openai").lower()
    if provider == "openai":
        if not OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing (env var).")
        llm = OpenAIClient(OPENAI_MODEL, OPENAI_KEY, TEMPERATURE, system_prompt)
    elif provider == "ollama":
        llm = OllamaClient(OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE, system_prompt)
    else:
        raise RuntimeError(f"Unsupported llm_provider: {LLM_PROVIDER}")

    total_lines = len(records)

    buffer: T.List[dict] = []
    processed = 0

    with tqdm(total=total_lines, desc="Classifying", unit="rec") as pbar:
        for rec in records:
            rid = rec.get("id")
            current_dataset = detect_dataset(rid) if rid else "Unknown"
            pbar.set_postfix(dataset=current_dataset)

            if not rid:
                out_rec = dict(rec)
                out_rec["Question Type"] = {"Intent": ["INTENT_9z. Others"], "Form": ["FORM_9z. Others"]}
                out_rec["_classification_error"] = "missing_id"
                buffer.append(out_rec)
            else:
                # -------------------- SPECIAL HANDLING FOR ClimaQA --------------------
                if rid.startswith("ClimaQA_Gold") or rid.startswith("ClimaQA_Silver"):
                    # 1) Detect question type label & forced Form (taxonomy-complete)
                    qtype_label, forced_form = _climaqa_subtype_and_form(rid)

                    # 2) Prefix english question type to text before sending to LLM
                    base_text = rec.get("text", "") or ""
                    send_text = f"{qtype_label}: {base_text}".strip()

                    # 3) Let LLM decide Intent (taxonomy-complete per prompt); ignore its Form
                    try:
                        qt_from_llm = llm.classify(send_text, max_retries=MAX_RETRIES)
                    except Exception as e:
                        qt_from_llm = {"intent": ["INTENT_9z. Others"], "form": ["FORM_9z. Others"], "_error": str(e)}

                    # 4) Overwrite Form with our forced Form; keep LLM Intent
                    out_rec = dict(rec)
                    out_rec["Question Type"] = {
                        "Intent": qt_from_llm.get("intent", []),
                        "Form": [forced_form],
                    }
                    if "_error" in qt_from_llm:
                        out_rec["_classification_error"] = qt_from_llm["_error"]
                    buffer.append(out_rec)

                else:
                    # -------------------- Original path: prefix mapping or send to LLM ---
                    mapped = prefix_mapping(rid)
                    if mapped is not None:
                        qt = mapped
                    else:
                        text = rec.get("text", "")
                        try:
                            qt = llm.classify(text, max_retries=MAX_RETRIES)
                        except Exception as e:
                            qt = {"intent": ["INTENT_9z. Others"], "form": ["FORM_9z. Others"], "_error": str(e)}

                    out_rec = dict(rec)
                    out_rec["Question Type"] = {
                        "Intent": qt.get("intent", []),
                        "Form": qt.get("form", []),
                    }
                    if "_error" in qt:
                        out_rec["_classification_error"] = qt["_error"]
                    buffer.append(out_rec)

            # periodic flush
            if len(buffer) >= FLUSH_INTERVAL:
                write_jsonl(out_path, buffer, mode="a")
                processed += len(buffer)
                buffer.clear()

            pbar.update(1)

    # Final flush
    if buffer:
        write_jsonl(out_path, buffer, mode="a")
        processed += len(buffer)

    print(f"[Done] Wrote: {out_path} | processed={processed} / total={total_lines}")


if __name__ == "__main__":
    main()
