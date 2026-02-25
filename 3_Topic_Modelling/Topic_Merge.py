from __future__ import annotations
import csv
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from tqdm.auto import tqdm

# Sentence-BERT
from sentence_transformers import SentenceTransformer, util

# Qwen3 embeddings (optional)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# OpenAI client
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "Configs.yaml").open("r", encoding="utf-8") as fp:
        root = yaml.safe_load(fp) or {}
    return root.get("3_Topic_Modelling", {})

CFG = load_cfg()
MCFG = CFG.get("Topic_Merge", {})

SUBJECT = MCFG.get("subject", "")
SUB_WORDS = MCFG.get("sub_words", 4)
EXPLAIN_WORDS = MCFG.get("explanation_words", 20)
SUBJECT_SLUG = SUBJECT.replace(" ", "_") if SUBJECT else ""
LOG_DIR = Path(MCFG.get("log_dir", f"Logs/{SUBJECT_SLUG}_20250428_204210")).expanduser()
TOPICS_DIR = LOG_DIR / "Topics"
TREE_PATH = LOG_DIR / "Topics_Tree.jsonl"

EMBED_MODEL = MCFG.get("embedding_model", "all-MiniLM-L6-v2")
BATCH_SIZE = int(MCFG.get("batch_size", 30))
MEAN_THRESH = float(MCFG.get("mean_threshold", 0.35))
MAX_THRESH = float(MCFG.get("max_threshold", 0.55))
STOP_MODE = MCFG.get("stop_criteria", "both").lower()   # inactivity|spread_mean|spread_max|both
TEMPERATURE = float(MCFG.get("temperature", 0.3))
MAX_RETRIES: int = MCFG.get("max_retries", 3)

OPENAI_MODEL = MCFG.get("openai_model", "gpt-4o")
OPENAI_KEY = MCFG.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
PROMPT_MERGE_PATH = Path(MCFG.get("merge_prompt_path", "Prompts/3_Merge_Topics.txt"))

LOCKED_TOPIC_NAMES = { "Irrelevant Data" }

def is_locked_topic(t: "Topic") -> bool:
    return (t.topic or "").strip() in LOCKED_TOPIC_NAMES

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def load_instructions() -> str:
    if PROMPT_MERGE_PATH.exists():
        txt = PROMPT_MERGE_PATH.read_text(encoding="utf-8")
        txt = txt.replace("{subject}", SUBJECT).replace("{n}", str(SUB_WORDS)).replace("{m}", str(EXPLAIN_WORDS))
        return txt.strip()

SYSTEM_INSTRUCTIONS = load_instructions()

# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lens]

class Qwen3Embedder:
    def __init__(self, model_name: str, device: str | torch.device = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = 8192

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        out_list = []
        for i in range(0, len(texts), batch_size):
            part = texts[i:i+batch_size]
            batch = self.tokenizer(part, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out = self.model(**batch)
                pooled = last_token_pool(out.last_hidden_state, batch["attention_mask"])
                normed = F.normalize(pooled, p=2, dim=1)
            out_list.append(normed.cpu().numpy())
        return np.vstack(out_list)

def make_embedder(name: str) -> Any:
    if name.lower().startswith("qwen3://"):
        model_id = name.split("://", 1)[1]
        return Qwen3Embedder(model_id)
    else:
        return SentenceTransformer(name)

embedder = make_embedder(EMBED_MODEL)

# ---------------------------------------------------------------------------
# OpenAI interaction
# ---------------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_KEY)

def ask_gpt_merge(
    parent: "Topic",
    candidates: List["Topic"],
) -> Tuple[str, str, List[str]]:
    """
    Input to LLM:
      - JSON array where each object has: id (string row index), topic, explanation
      - id '1' is the parent; candidates are '2'.. up to batch size+1
    Output expected from LLM:
      - JSON object: {"merged_ids": ["2","5",...], "parent_topic":"...", "parent_explanation":"..."}
    """
    # Build row-indexed payload for the model (ids are simple strings "1","2",...)
    arr = [
        {"id": "1", "topic": parent.topic, "explanation": parent.explanation}
    ] + [
        {"id": str(i + 2), "topic": c.topic, "explanation": c.explanation}
        for i, c in enumerate(candidates)
    ]
    payload = json.dumps(arr, ensure_ascii=False)
    valid_ids = {x["id"] for x in arr}
    parent_row_id = "1"

    # Map back from row id → candidate object
    id2obj = {str(i + 2): c for i, c in enumerate(candidates)}
    # --- LLM call ---
    for attempt in range(MAX_RETRIES):
        resp = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=payload,
            # temperature=TEMPERATURE,
            reasoning={
                "effort": "minimal"
            }
        )
        txt = (resp.output_text or "").strip()
        try:
            data = json.loads(txt)
            new_topic = (data.get("parent_topic") or parent.topic).strip()
            new_expl  = (data.get("parent_explanation") or parent.explanation).strip()

            merged_ids = [i for i in data.get("merged_ids", []) if isinstance(i, str)]
            # Validate & exclude parent
            merged_ids = [i for i in merged_ids if i in valid_ids and i != parent_row_id]

            rowid2id = {str(i + 2): c.id for i, c in enumerate(candidates)}
            merged_true_ids = [rowid2id[i] for i in merged_ids if i in rowid2id]

            return new_topic, new_expl, merged_true_ids
        except Exception:
            print(f"[RETRY-{attempt+1}] Invalid JSON from LLM; retrying...")
            time.sleep(1.5)

    print("[FALLBACK] Using parent unchanged.")
    return parent.topic, parent.explanation, []


# ---------------------------------------------------------------------------
# Data structure and IO
# ---------------------------------------------------------------------------

class Topic:
    __slots__ = ("id", "topic", "explanation", "count", "emb", "merged")
    def __init__(self, _id: str | None, topic: str, explanation: str, count: int, emb: np.ndarray):
        self.id = _id                  # may be None for next-level parents until assigned
        self.topic = topic
        self.explanation = explanation
        self.count = count
        self.emb = emb
        self.merged = False

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _level_from_csv_path(p: Path) -> int:
    m = re.search(r"Topics_(\d+)\\.csv$", p.name)
    return int(m.group(1)) if m else 1


def load_level(csv_path: Path, embedder: Any) -> List[Topic]:
    """Load a level CSV and return Topic objects whose `.id` matches the CSV.
    If the CSV lacks the id column (3-col), it will be rewritten on disk with `id`.
    """
    level = _level_from_csv_path(csv_path)
    rows: List[Tuple[str, str, int, str]] = []  # (topic, expl, count, id)
    needs_id_writeback = False

    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        row_idx = 0
        for r in reader:
            if not r:
                continue
            row_idx += 1
            if len(r) >= 4:
                _id, topic, expl, cnt = r[0], r[1], r[2], int(r[3])
            else:
                _id, topic, expl, cnt = f"L{level}_{row_idx}", r[0], r[1], int(r[2])
                needs_id_writeback = True
            rows.append((topic, expl, cnt, _id))

    if needs_id_writeback:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            wr = csv.writer(fp)
            for topic, expl, cnt, _id in rows:
                wr.writerow([_id, topic, expl, cnt])
        print(f"[INFO] Completed missing id column for {csv_path.name}")

    # Collapse identical (topic, explanation) within this level, preserving a representative id
    tmp: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for topic, expl, cnt, _id in rows:
        norm_key = (_norm_text(topic), _norm_text(expl))
        if norm_key not in tmp:
            tmp[norm_key] = {
                "cnt": cnt,
                "rep_id": _id,  # keep the first seen id as representative
                "orig_topic": (topic or "").strip(),
                "orig_expl": (expl or "").strip(),
            }
        else:
            tmp[norm_key]["cnt"] += cnt
            # keep the first seen rep_id; other duplicate ids are omitted by design

    labels = [f"{v['orig_topic']}: {v['orig_expl']}" for v in tmp.values()]  # use topic name + explanation for embedding similarity
    # labels = [v["orig_topic"] for v in tmp.values()] # only use topic name for embedding similarity

    if isinstance(embedder, SentenceTransformer):
        embs = embedder.encode(labels, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    else:
        embs = embedder.encode(labels, batch_size=64)

    topics: List[Topic] = []
    for (data, vec) in zip(tmp.values(), embs):
        topics.append(Topic(
            data["rep_id"],
            topic=data["orig_topic"],
            explanation=data["orig_expl"],
            count=int(data["cnt"]),
            emb=vec,
        ))
    return topics


def save_level(csv_path: Path, topics: List[Topic]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        wr = csv.writer(fp)
        for t in topics:
            assert t.id is not None, "Attempting to save a topic without an assigned id"
            wr.writerow([t.id, t.topic, t.explanation, t.count])


def append_tree(lines: List[Dict]) -> None:
    TREE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TREE_PATH.open("a", encoding="utf-8") as fp:
        for l in lines:
            json.dump(l, fp, ensure_ascii=False)
            fp.write("\n")

# ---------------------------------------------------------------------------
# Similarity and stop criteria
# ---------------------------------------------------------------------------

def similarity_stats(vec: np.ndarray, others: np.ndarray) -> Tuple[float, float, float]:
    if others.size == 0:
        return 0.0, 0.0, 0.0
    sims = util.pytorch_cos_sim(vec, others).flatten().cpu().numpy()
    return float(sims.max()), float(sims.min()), float(sims.mean())


def inter_topic_spread(topics: List[Topic]) -> Tuple[float, float]:
    if len(topics) < 2:
        return 0.0, 0.0
    mat = np.stack([t.emb for t in topics])
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    sims = util.pytorch_cos_sim(mat, mat).cpu().numpy()
    triu = sims[np.triu_indices_from(sims, k=1)]
    return float(triu.mean()), float(triu.max())

# ---------------------------------------------------------------------------
# End-of-round deduplication for identical text
# ---------------------------------------------------------------------------

def dedup_topics(items: List[Topic]) -> List[Topic]:
    """Merge topics with identical (topic, explanation). IDs are not assigned here.
    The caller will assign stable IDs based on next level and row order.
    """
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in items:
        key = (_norm_text(t.topic), _norm_text(t.explanation))
        if key not in buckets:
            buckets[key] = {
                "topic": t.topic, "expl": t.explanation,
                "cnt": t.count, "emb": t.emb * t.count,
            }
        else:
            buckets[key]["cnt"] += t.count
            buckets[key]["emb"] += t.emb * t.count

    out: List[Topic] = []
    for v in buckets.values():
        cnt = max(int(v["cnt"]), 1)
        emb = v["emb"] / float(cnt)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        out.append(Topic(None, v["topic"], v["expl"], cnt, emb))
    out.sort(key=lambda x: x.count, reverse=True)
    return out

def _append_pending_edge_self(level: int, t: Topic, pending_edges: List[Dict[str, Any]]) -> None:
    """record “this turn's parent's next turn's parent is itself."""
    parent_key = (_norm_text(t.topic), _norm_text(t.explanation))
    pending_edges.append({
        "level": level + 1,
        "parent_key": parent_key,
        "parent_topic": t.topic,
        "parent_explanation": t.explanation,
        "children_ids": [t.id],
        "children_topics": [t.topic],
    })


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    level = 1
    cur_csv = TOPICS_DIR / f"Topics_{level}.csv"
    if not cur_csv.exists():
        print("Topics_1.csv not found. Run preliminary step first.")
        sys.exit(1)

    while True:
        merges_this_round = 0
        topics_merged_this_round = 0

        topics = load_level(cur_csv, embedder)
        topics.sort(key=lambda t: t.count, reverse=True)

        any_merge = False
        next_topics: List[Topic] = []
        # We temporarily collect edges referencing parent *keys*. We'll resolve to IDs after dedup & ID assignment.
        pending_edges: List[Dict[str, Any]] = []

        itr = tqdm(topics, desc=f"Round {level} merging", unit="topic")
        for parent in itr:
            if parent.merged:
                continue

            if is_locked_topic(parent):
                parent.merged = True
                next_topics.append(parent)
                _append_pending_edge_self(level, parent, pending_edges)
                print(f"\nSkip Irrelevant Data.")
                continue

            unmerged = [t for t in topics if not t.merged and t is not parent and not is_locked_topic(t)]
            if not unmerged:
                parent.merged = True
                next_topics.append(parent)
                _append_pending_edge_self(level, parent, pending_edges)
                continue

            others_vec = np.stack([u.emb for u in unmerged])
            max_s, min_s, mean_s = similarity_stats(parent.emb, others_vec)
            tqdm.write(f"\n[SIM] '{parent.topic[:50]}.' max={max_s:.3f} min={min_s:.3f} mean={mean_s:.3f}")

            sims = util.pytorch_cos_sim(parent.emb, others_vec).flatten().cpu().numpy()
            # --- Adaptive k: take the smaller of the preset BATCH_SIZE and 1/10 of total topics (at least 1)
            total_topics = len(topics)
            k_by_total = max(1, total_topics // 10)
            k = min(BATCH_SIZE, k_by_total)
            top_n = min(k, len(unmerged))

            top_idx = np.argsort(-sims)[:top_n]
            batch = [unmerged[i] for i in top_idx]

            new_topic, new_expl, merged_ids = ask_gpt_merge(parent, batch)

            id2obj = {o.id: o for o in batch}
            merge_objs: List[Topic] = [parent]
            for mid in merged_ids:
                obj = id2obj.get(mid)
                if obj and (obj not in merge_objs):
                    merge_objs.append(obj)

            if len(merge_objs) > 1:
                any_merge = True
                merges_this_round += 1
                topics_merged_this_round += len(merge_objs)

                child_names = [o.topic for o in merge_objs]  # includes parent
                preview = ", ".join([f"'{n}'" for n in child_names[:5]])
                if len(child_names) > 5:
                    preview += f", … +{len(child_names) - 5} more"

                tqdm.write(f""
                           f"\n[MERGE][L{level}] {len(merge_objs)} topics → '{new_topic[:60]}' | children: {preview}\n")

            weights = np.array([o.count for o in merge_objs], dtype=float)
            vecs = np.stack([o.emb for o in merge_objs])
            combined_emb = np.average(vecs, axis=0, weights=weights)
            combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)
            combined_count = int(weights.sum())

            new_obj = Topic(None, new_topic, new_expl, combined_count, combined_emb)
            next_topics.append(new_obj)

            for o in merge_objs:
                o.merged = True

            # record edge to resolve after dedup & ID assignment
            children_ids = [o.id for o in merge_objs]  # include parent in the tree
            parent_key = (_norm_text(new_obj.topic), _norm_text(new_obj.explanation))
            pending_edges.append({
                "level": level + 1,
                "parent_key": parent_key,
                "parent_topic": new_obj.topic,
                "parent_explanation": new_obj.explanation,
                "children_ids": children_ids,
                "children_topics": [o.topic for o in merge_objs],  # include parent in the tree
            })

        # ---- check all this turn's child topics is covered by a path as the children
        orig_ids = {t.id for t in topics}
        covered_ids = set()
        for e in pending_edges:
            for cid in e["children_ids"]:
                if cid:
                    covered_ids.add(cid)

        missing_ids = orig_ids - covered_ids
        if missing_ids:
            id2obj_full = {t.id: t for t in topics}
            for mid in missing_ids:
                t = id2obj_full[mid]
                next_topics.append(t)  # take into next turn
                _append_pending_edge_self(level, t, pending_edges)  # add an edge (itself)
            print(f"[FIXUP][L{level}] added {missing_ids} self-edges for uncovered topics.")

        # dedup parents and then assign next-level IDs deterministically
        next_topics = dedup_topics(next_topics)
        # Assign ids as L{level+1}_{row}
        for i, t in enumerate(next_topics, 1):
            t.id = f"L{level+1}_{i}"

        # Build parent_key -> parent_id map (after dedup)
        key2id = {(_norm_text(t.topic), _norm_text(t.explanation)): t.id for t in next_topics}

        # materialize tree with matching ids
        tree_lines: List[Dict[str, Any]] = []
        for e in pending_edges:
            pid = key2id.get(e["parent_key"])  # dedup may fold multiple parents into one id
            if not pid:
                # Shouldn't happen, but guard anyway
                continue
            tree_lines.append({
                "level": e["level"],
                "parent_topic": e["parent_topic"],
                "parent_explanation": e["parent_explanation"],
                "children_topics": e["children_topics"],
                "parent_id": pid,
                "children_ids": e["children_ids"],
            })

        append_tree(tree_lines)

        next_csv = TOPICS_DIR / f"Topics_{level+1}.csv"
        save_level(next_csv, next_topics)

        mean_spread, max_spread = inter_topic_spread(next_topics)
        stop_inact = (not any_merge) and STOP_MODE in ("inactivity", "both")
        stop_mean = (mean_spread <= MEAN_THRESH) and STOP_MODE in ("spread_mean", "both")
        stop_max = (max_spread <= MAX_THRESH) and STOP_MODE in ("spread_max", "both")

        print(f"[ROUND {level}] → {len(next_topics)} topics | any_merge={any_merge} | "
              f"mean_spread={mean_spread:.3f} | max_spread={max_spread:.3f}")
        avg = (topics_merged_this_round / merges_this_round) if merges_this_round else 0.0
        print(f"[ROUND {level} SUMMARY] merges={merges_this_round} | "
              f"topics_merged={topics_merged_this_round} | avg_per_merge={avg:.2f}")

        if stop_inact or stop_mean or stop_max:
            final_csv = TOPICS_DIR / "Topics_Final.csv"
            save_level(final_csv, next_topics)
            print(f"[DONE] Saved final topics to {final_csv.relative_to(LOG_DIR)}")
            break

        level += 1
        cur_csv = next_csv

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        print(f"Elapsed {time.time() - t0:.1f}s")
