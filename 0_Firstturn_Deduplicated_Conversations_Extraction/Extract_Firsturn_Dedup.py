from __future__ import annotations

import hashlib
import sys
import yaml
from pathlib import Path
from typing import Dict, Iterable


# --------------------------------------------------------------------------- #
# Fast JSON if available ----------------------------------------------------- #
try:  # pragma: no cover
    import orjson as _json

    def _dumps(obj):
        return _json.dumps(obj, option=_json.OPT_NON_STR_KEYS).decode()

    _loads = _json.loads
except ModuleNotFoundError:  # fallback to stdlib json
    import json as _json

    _dumps = _json.dumps
    _loads = _json.loads

# --------------------------------------------------------------------------- #
# Configuration constants ---------------------------------------------------- #
CLI_CFG = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
DEFAULT_CFG = Path(__file__).resolve().parent.parent / "Configs.yaml"
CFG_PATH = CLI_CFG or DEFAULT_CFG
SECTION = "0_Firstturn_Deduplicated_Conversations_Extraction"

if not CFG_PATH.is_file():
    sys.exit(f"Config file not found: {CFG_PATH}")

with CFG_PATH.open("r", encoding="utf-8") as f:
    ROOT_CFG = yaml.safe_load(f) or {}

CFG = ROOT_CFG.get(SECTION, {})

DATA_DIR = Path(CFG.get("data_dir", "Data"))
INPUT_FILE = CFG.get("input_file", "WildChat_Original.jsonl")
FIRSTTURN_FILE = CFG.get("firstturn_file", "WildChat_Firstturn.jsonl")
DEDUP_FILE = CFG.get("dedup_file", "WildChat_Firstturn_dedup.jsonl")

ENCODING = CFG.get("encoding", "utf-8")
CONVERSATION_KEY = CFG.get("conversation_key", "conversation")
USER_ROLE = CFG.get("user_role_value", "user")
ASSISTANT_ROLE = CFG.get("assistant_role_value", "assistant")
TEXT_FIELD = CFG.get("text_field", "content")

DEDUP_ON_EMPTY = bool(CFG.get("dedup_on_empty_user_question", False))
STRIP_WS = bool(CFG.get("strip_whitespace", True))
HASH_KEYS = bool(CFG.get("dedup_hash_questions", True))
ONLY_USER = bool(CFG.get("only_user", False))

DEFAULT_KEEP = [
    "conversation_hash",
    "text",
    "model",
    "language",
    "state",
    "country",
    "toxic",
]
KEEP_FIELDS = CFG.get("keep_fields", DEFAULT_KEEP)

# --------------------------------------------------------------------------- #
# Helper functions ----------------------------------------------------------- #

def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _first_turn(msgs: Iterable[Dict]) -> tuple[str, str]:
    """Return (user_question_raw, assistant_answer_raw)."""
    user_q = ""
    assistant_a = ""

    for m in msgs:
        role = m.get("role")
        if not user_q and role == USER_ROLE:
            user_q = m.get(TEXT_FIELD, "")
        elif user_q and role == ASSISTANT_ROLE:
            assistant_a = m.get(TEXT_FIELD, "")
            break
    return user_q, assistant_a


# --------------------------------------------------------------------------- #
# Main streaming loop -------------------------------------------------------- #

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    in_path = DATA_DIR / INPUT_FILE
    if not in_path.is_file():
        sys.exit(f"Input file not found: {in_path}")

    ft_path = DATA_DIR / FIRSTTURN_FILE
    dd_path = DATA_DIR / DEDUP_FILE

    # dedup mapping: user_key ➜ first formatted text seen
    seen_questions: dict[str, str] = {}

    # counters
    orig_cnt = first_cnt = dedup_cnt = 0
    user_empty_cnt = both_empty_cnt = 0
    dup_user_only_cnt = dup_user_assistant_cnt = 0

    with (
        in_path.open("r", encoding=ENCODING) as fin,
        ft_path.open("w", encoding=ENCODING, buffering=1) as fout_ft,
        dd_path.open("w", encoding=ENCODING, buffering=1) as fout_dd,
    ):
        for line in fin:
            if not line.strip():
                continue
            orig_cnt += 1
            rec = _loads(line)

            # ---- extract first turn ---- #
            user_raw, assistant_raw = _first_turn(rec.get(CONVERSATION_KEY, []))
            user_stripped = user_raw.strip()
            assistant_stripped = assistant_raw.strip()

            if not user_stripped and not assistant_stripped:
                both_empty_cnt += 1
                continue  # skip entirely

            if not ONLY_USER:
              formatted_text = f"user: {user_stripped}; assistant: {assistant_stripped}"
            if ONLY_USER:
                if not user_stripped:
                    user_empty_cnt += 1
                    continue  # skip entirely
                formatted_text = f"{user_stripped}"

            # build minimal record
            out_rec = {fld: rec.get(fld) for fld in KEEP_FIELDS if fld != "text"}
            out_rec["text"] = formatted_text

            # write to first‑turn file
            fout_ft.write(_dumps(out_rec) + "\n")
            first_cnt += 1

            # --- deduplication logic --- #
            if not user_stripped:
                # user empty but assistant non‑empty
                user_empty_cnt += 1
                dedup_cnt += 1
                fout_dd.write(_dumps(out_rec) + "\n")
                continue

            key_norm = user_stripped if not STRIP_WS else user_stripped
            key = _sha(key_norm) if HASH_KEYS else key_norm

            if not ONLY_USER:
                prev_text = seen_questions.get(key)
                if prev_text is not None:
                    # duplicate user question
                    if formatted_text == prev_text:
                        dup_user_assistant_cnt += 1
                    else:
                        dup_user_only_cnt += 1
                    # duplicates are not written to dedup file
                    continue

            if ONLY_USER:
                prev_text = seen_questions.get(key)
                if prev_text is not None:
                    # duplicate user question
                    if formatted_text == prev_text:
                        dup_user_only_cnt += 1
                    continue

            # first occurrence
            seen_questions[key] = formatted_text
            dedup_cnt += 1
            fout_dd.write(_dumps(out_rec) + "\n")

    # ---- summary ---- #
    print("Extraction finished ✅\n")
    print(f"Original conversations             : {orig_cnt}")
    print(f"First‑turn conversations written   : {first_cnt}")
    print(f"After deduplication (written)      : {dedup_cnt}\n")
    print("Additional counts:")
    print(f"  user empty, assistant present    : {user_empty_cnt}")
    print(f"  both user & assistant empty      : {both_empty_cnt}")
    print(f"  duplicate user only              : {dup_user_only_cnt}")
    print(f"  duplicate user + assistant       : {dup_user_assistant_cnt}\n")
    print("Files written:")
    print(f"  • {ft_path}")
    print(f"  • {dd_path}")


if __name__ == "__main__":
    main()
