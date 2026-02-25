from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml


def load_cfg() -> dict:
    """Load the top-level config section `3_Topic_Modelling` from Configs.yaml.
    The file is looked up at the repository root (one level above this file's parent).
    """
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "Configs.yaml").open("r", encoding="utf-8") as fp:
        root = yaml.safe_load(fp) or {}
    return root.get("3_Topic_Modelling", {})

CFG = load_cfg()

MCFG = CFG.get("Topic_Merge", {})

# Parameters for THIS script live in `Topic_Trace` section
TCFG = CFG.get("Transitional_Final_Topics", {})

# Core paths
LOG_DIR = Path(TCFG.get("log_dir", "")).expanduser()
TOPICS_DIR = LOG_DIR / "Topics"
TREE_PATH = LOG_DIR / TCFG.get("topics_tree", "Topics_Tree.jsonl")
DATA_WITH_TOPIC = LOG_DIR / TCFG.get("data_with_topic", "All_Data_with_Topic.jsonl")
DATA_WITH_FINAL = LOG_DIR / TCFG.get("data_with_final_topic", "All_Data_with_Final_Topic.jsonl")

# CSV file names
SOURCE_TOPICS_CSV = TOPICS_DIR / TCFG.get("source_topics_csv", "Topics_1.csv")
FINAL_TOPICS_CSV = TOPICS_DIR / TCFG.get("final_topics_csv", "Topics_Final.csv")
CSV_DELIMITER = str(TCFG.get("csv_delimiter", ","))

# Field & behavior tuning
INITIAL_FIELD_CANDIDATES: List[str] = TCFG.get(
    "initial_field_candidates",
    ["Initial Topics", "Initial_Topics", "Initial_Topic"],
)
LOCKED_TOPIC_NAMES: Set[str] = set(TCFG.get("locked_topic_names", ["Irrelevant Data"]))

# =============================================================================
# CSV readers and catalogs
# =============================================================================

class TopicsCSV:
    """Loader for a headerless `Topics_k.csv`.

    Expected columns (no header):
    0 = id (e.g., Lk_row), 1 = topic, 2 = explanation, 3 = count (optional)

    This class exposes:
      - `idx_to_id` (for 1-based position → id mapping)
      - `id_to_topic` (id → (topic, explanation))
    """

    def __init__(self, csv_path: Path, delimiter: str = ","):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing topics CSV: {csv_path}")
        self.csv_path = csv_path
        self.delimiter = delimiter
        self.idx_to_id: List[str] = []
        self.id_to_topic: Dict[str, Tuple[str, str]] = {}
        self._load()

    def _load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp, delimiter=self.delimiter)
            for row in reader:
                if not row:
                    continue
                # Ensure at least 3 columns are available
                if len(row) < 3:
                    row = row + [""] * (3 - len(row))
                _id = (row[0] or "").strip()
                topic = (row[1] or "").strip()
                expl = (row[2] or "").strip()
                if not _id:
                    continue
                self.idx_to_id.append(_id)
                self.id_to_topic[_id] = (topic, expl)

    def id_by_1based_index(self, idx: int) -> Optional[str]:
        """Return the id at 1-based position `idx`, or None if out of range."""
        if 1 <= idx <= len(self.idx_to_id):
            return self.idx_to_id[idx - 1]
        return None


class AllTopicsCatalog:
    """Aggregate `id → (topic, explanation)` from **all** `Topics_*.csv` under `TOPICS_DIR`.

    This is used to resolve final names/explanations unambiguously, even if the final node
    is a previous-level topic (e.g., L1 for "Irrelevant Data").
    """

    def __init__(self, topics_dir: Path, delimiter: str = ","):
        self.id_to_topic_expl: Dict[str, Tuple[str, str]] = {}
        for p in sorted(topics_dir.glob("Topics_*.csv")):
            with p.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    if len(row) < 3:
                        row = row + [""] * (3 - len(row))
                    _id = (row[0] or "").strip()
                    topic = (row[1] or "").strip()
                    expl = (row[2] or "").strip()
                    if _id:
                        self.id_to_topic_expl[_id] = (topic, expl)

    def get(self, _id: str) -> Optional[Tuple[str, str]]:
        return self.id_to_topic_expl.get(_id)


# =============================================================================
# Tree loader and climb logic
# =============================================================================

class ParentMap:
    """Build a child→parent map from `Topics_Tree.jsonl`.

    Accepts multiple schema variants for compatibility:
      - New: `parent_id` / `children_ids`
      - Old: `parent_tid` / `children_tids`
      - Very old: `parent` / `children`
    """

    def __init__(self, tree_path: Path):
        if not tree_path.exists():
            raise FileNotFoundError(f"Missing tree: {tree_path}")
        self.tree_path = tree_path
        self.child_to_parent: Dict[str, str] = {}
        self._load()

    @staticmethod
    def _first_str(node: dict, *keys: str) -> Optional[str]:
        for k in keys:
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    @staticmethod
    def _first_list_of_str(node: dict, *keys: str) -> List[str]:
        for k in keys:
            v = node.get(k)
            if isinstance(v, list) and v:
                out = []
                for x in v:
                    s = (str(x) if x is not None else "").strip()
                    if s:
                        out.append(s)
                return out
        return []

    def _load(self) -> None:
        with self.tree_path.open("r", encoding="utf-8") as fp:
            for ln in fp:
                if not ln.strip():
                    continue
                node = json.loads(ln)
                parent = self._first_str(node, "parent_id", "parent_tid", "parent")
                if not parent:
                    # Ignore malformed lines without a parent
                    continue
                children = self._first_list_of_str(node, "children_ids", "children_tids", "children")
                for c in children:
                    self.child_to_parent[c] = parent


# ANSI colors for errors
_RED = "\033[31m"
_END = "\033[0m"

# Parse an ID like L12_34 → level = 12
_ID_RE = re.compile(r"^[Ll](\d+)_\d+$")

def parse_level(tid: str) -> Optional[int]:
    m = _ID_RE.match(tid or "")
    return int(m.group(1)) if m else None


def climb_to_root(start_id: str, pmap: ParentMap) -> Tuple[str, bool]:
    """Climb child→parent until no parent or a cycle is found.

    Returns:
      (reached_id, moved)
      - `reached_id`: the last reachable id
      - `moved`: whether we actually followed at least one parent edge
    """
    seen = {start_id}
    cur = start_id
    moved = False
    while True:
        nxt = pmap.child_to_parent.get(cur)
        if not nxt or nxt in seen:
            return cur, moved
        cur = nxt
        moved = True
        seen.add(cur)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Validate required files
    for p in [DATA_WITH_TOPIC, TREE_PATH, SOURCE_TOPICS_CSV, FINAL_TOPICS_CSV]:
        if not Path(p).exists():
            print(f"{_RED}ERROR: Missing file {p}{_END}", file=sys.stderr)
            sys.exit(1)

    # Load CSVs and catalogs
    src_topics = TopicsCSV(SOURCE_TOPICS_CSV, CSV_DELIMITER)
    final_topics = TopicsCSV(FINAL_TOPICS_CSV, CSV_DELIMITER)
    final_id_set: Set[str] = set(final_topics.id_to_topic.keys())

    # Determine the highest level (from the IDs present in the final CSV)
    final_levels = {lvl for _id in final_id_set if (lvl := parse_level(_id)) is not None}
    max_final_level = max(final_levels) if final_levels else None

    catalog = AllTopicsCatalog(TOPICS_DIR, CSV_DELIMITER)
    pmap = ParentMap(TREE_PATH)

    def resolve_record(rec: dict, line_no: int) -> dict:
        # 1) Pick the first present initial-topics field with an int list value
        init_idxs: Optional[Iterable[int]] = None
        for k in INITIAL_FIELD_CANDIDATES:
            v = rec.get(k)
            if isinstance(v, list) and all(isinstance(x, int) for x in v):
                init_idxs = v
                break
        if init_idxs is None:
            print(
                f"{_RED}ERROR[L{line_no}]: no initial topic field among {INITIAL_FIELD_CANDIDATES}{_END}",
                file=sys.stderr,
            )
            rec["Final_Topic_IDs"] = []
            rec["Final_Topics"] = []
            rec["Final_Topics_Explanation"] = []
            return rec

        finals: List[str] = []

        for idx in init_idxs:
            if not isinstance(idx, int):
                print(f"{_RED}ERROR[L{line_no}]: initial index not int -> {idx}{_END}", file=sys.stderr)
                continue

            # 2) Map 1-based row index from Topics_1.csv to its id (e.g., L1_23)
            start_id = src_topics.id_by_1based_index(idx)
            if not start_id:
                print(
                    f"{_RED}ERROR[L{line_no}]: initial index {idx} out of range (1..{len(src_topics.idx_to_id)}){_END}",
                    file=sys.stderr,
                )
                continue

            # 3) Climb child→parent until no parent is found
            reached, moved = climb_to_root(start_id, pmap)

            # 4) Acceptance rules
            ok = False
            # 4.1 In the explicit final set
            if reached in final_id_set:
                ok = True
            else:
                # 4.2 Special case: L1 locked topics without any move
                if not moved and parse_level(start_id) == 1:
                    pair = catalog.get(start_id)
                    if pair and (pair[0] in LOCKED_TOPIC_NAMES):
                        reached = start_id
                        ok = True
                # 4.3 Highest level acceptance: if at the highest level (e.g., L30), treat as final
                if not ok and max_final_level is not None:
                    lvl = parse_level(reached)
                    if lvl is not None and lvl == max_final_level:
                        ok = True
                # 4.4 Otherwise warn, but keep `reached` in the output for debugging
                if not ok:
                    print(
                        f"{_RED}ERROR[L{line_no}]: start {start_id} did not reach final set (stopped at {reached}){_END}",
                        file=sys.stderr,
                    )

            if reached not in finals:
                finals.append(reached)

        # 5) Map ids → (name, explanation) via the catalog
        names: List[str] = []
        expls: List[str] = []
        for tid in finals:
            pair = catalog.get(tid)
            if pair is None:
                print(
                    f"{_RED}ERROR[L{line_no}]: id {tid} missing in any Topics_*.csv (name/expl lookup failed){_END}",
                    file=sys.stderr,
                )
                names.append(tid)
                expls.append("")
            else:
                names.append(pair[0] or tid)
                expls.append(pair[1] or "")

        rec["Final_Topic_IDs"] = finals
        rec["Final_Topics"] = names
        rec["Final_Topics_Explanation"] = expls
        return rec

    # Stream in → out
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_WITH_TOPIC.open("r", encoding="utf-8") as fin, DATA_WITH_FINAL.open("w", encoding="utf-8") as fout:
        for i, ln in enumerate(fin, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception as e:
                print(f"{_RED}ERROR[L{i}]: invalid JSON — {e}{_END}", file=sys.stderr)
                continue
            out = resolve_record(rec, i)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {DATA_WITH_FINAL}")


if __name__ == "__main__":
    main()
